#!/usr/bin/env python3
"""Rerun the reader on a fixed subset of questions and keep its THINKING (reasoning_content).

Why: the harness only caches the final answer text; the error analysis (error-analysis-plan
2026-10-26) needs the reasoning. The cached prompt rows of the source arm are replayed
byte-for-byte through the same OpenAI-compatible endpoint with the harness's sampling
(temperature 0.6, top_p 0.95, top_k 20, max_tokens 20000, thinking on), and vLLM's qwen3
reasoning parser returns the thinking separately. The rerun is a second sample, so a question
can flip; the analysis should focus on questions wrong in both runs (see `score`).

Modes (run on the cluster, `~/e1` layout):
  select   choose the question groups from the waterfall files and write the id list
           A  waterfall stage 5* (read wrong)                         -> arm b1
           B  waterfall stage 3* and wrong (not retrieved / not in context) -> arm b1
           C  fusion vs B1 disagreements (r4_rule_fusion_complete.jsonl) -> arms b1 + fusion
           D  stage 6_correct, 12 questions stratified by type x domain, seed 0 -> arm b1
  run      --arm b1|fusion: call the reader; append to
           ~/e1/runs/think_<arm>_small/<dom>_small/thinking.jsonl (resumable)
  score    rule-score thinking.jsonl with the official qa_eval_metrics -> scores.jsonl,
           and print the reproduction table (original vs rerun) per group
  export-c per-question rows (without memory_context) of both arms for the C questions
Usage:
  python e1/oneoff/rerun_with_thinking.py select
  python e1/oneoff/rerun_with_thinking.py run --arm b1 --concurrency 6
  python e1/oneoff/rerun_with_thinking.py score
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from evaluation.qa_eval_metrics import eval_from_spec, extract_boxed_answer, is_unknown, score_to_bool  # noqa: E402

E1 = Path(os.environ.get("E1", "~/e1")).expanduser()
DOMS = ("web", "enterprise")
ARMS = {"b1": "b1_small", "fusion": "e2_fusion_small"}  # short name -> source arm folder
IDS_DEFAULT = E1 / "e2work/reports/error_analysis_ids.json"
D_SIZE = 12
D_SEED = 0


def reasoning_of(msg) -> str:
    for k in ("reasoning_content", "reasoning"):
        v = getattr(msg, k, None)
        if isinstance(v, str) and v:
            return v
    extra = getattr(msg, "model_extra", None) or {}
    for k in ("reasoning_content", "reasoning"):
        if isinstance(extra.get(k), str) and extra[k]:
            return extra[k]
    return ""


def think_dir(arm: str, dom: str) -> Path:
    return E1 / f"runs/think_{arm}_small/{dom}_small"


def load_ids(path: Path) -> dict:
    return json.loads(Path(path).expanduser().read_text())


# ----------------------------------------------------------------------------- select
def allocate(cells: dict[str, int], total: int) -> dict[str, int]:
    """At least one per non-empty cell, the rest proportional (largest remainder)."""
    cells = {k: v for k, v in cells.items() if v > 0}
    alloc = {k: 1 for k in cells}
    rest = total - len(alloc)
    assert rest >= 0, "more cells than samples"
    n = sum(cells.values())
    quota = {k: rest * v / n for k, v in cells.items()}
    for k in cells:
        alloc[k] += int(quota[k])
    left = total - sum(alloc.values())
    for k in sorted(cells, key=lambda k: quota[k] - int(quota[k]), reverse=True)[:left]:
        alloc[k] += 1
    return alloc


def select(args) -> None:
    W = {}
    for dom in DOMS:
        for line in open(E1 / f"e2work/rounds/waterfall/b1_small_{dom}.jsonl"):
            if line.strip():
                r = json.loads(line)
                W[r["question_id"]] = r
    A = sorted(q for q, r in W.items() if r["stage"].startswith("5"))
    B = sorted(q for q, r in W.items() if r["stage"].startswith("3") and not r["correct"])
    # D: 12 correct questions stratified by question_type x domain
    cells = defaultdict(list)
    for q, r in W.items():
        if r["stage"] == "6_correct" and r["correct"]:
            cells[(r["domain"], r["question_type"])].append(q)
    cells = {k: sorted(v) for k, v in cells.items() if k[1] != "errors-gotchas"}
    alloc = allocate({f"{d}|{t}": len(v) for (d, t), v in cells.items()}, D_SIZE)
    rng = random.Random(D_SEED)
    D = []
    for (d, t), v in sorted(cells.items()):
        D += sorted(rng.sample(v, alloc[f"{d}|{t}"]))
    # C: fusion vs B1 disagreements on rule-scored complete questions
    cpath = E1 / "e2work/reports/r4_rule_fusion_complete.jsonl"
    R = [json.loads(l) for l in open(cpath) if l.strip()]
    C_fixed = sorted(r["question_id"] for r in R if r["arm_correct"] and not r["ref_correct"])
    C_broke = sorted(r["question_id"] for r in R if r["ref_correct"] and not r["arm_correct"])
    meta = {}
    for dom in DOMS:
        for line in open(E1 / f"cache/meta/{dom}.jsonl"):
            if line.strip():
                r = json.loads(line)
                meta[r["question_id"]] = {"domain": dom, "question_type": r["question_type"]}
    by_id = {}
    for g, ids in (("A", A), ("B", B), ("C_fixed", C_fixed), ("C_broke", C_broke), ("D", D)):
        for q in ids:
            e = by_id.setdefault(q, {"groups": [], "arms": ["b1"], **meta[q],
                                     "b1_correct_original": W[q]["correct"] if q in W else None,
                                     "waterfall_stage": W[q]["stage"] if q in W else None})
            e["groups"].append(g)
            if g.startswith("C") and "fusion" not in e["arms"]:
                e["arms"].append("fusion")
    out = {
        "written": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "sources": {"waterfall": "e2work/rounds/waterfall/b1_small_{web,enterprise}.jsonl",
                    "C": str(cpath.relative_to(E1))},
        "rules": {"A": "waterfall stage starts with 5", "B": "waterfall stage starts with 3 and correct == false",
                  "C": "fusion fixed / broke vs B1 on rule-scored complete questions; both arms rerun",
                  "D": f"stage 6_correct, {D_SIZE} questions, >=1 per domain x type cell then largest remainder, "
                       f"random.Random({D_SEED}).sample on sorted ids, errors-gotchas excluded"},
        "D_allocation": alloc,
        "groups": {"A": A, "B": B, "C_fixed": C_fixed, "C_broke": C_broke, "D": D},
        "counts": {"A": len(A), "B": len(B), "C_fixed": len(C_fixed), "C_broke": len(C_broke), "D": len(D),
                   "unique_b1": len(by_id), "fusion": sum(1 for e in by_id.values() if "fusion" in e["arms"]),
                   "C_not_in_waterfall": sum(1 for q in C_fixed + C_broke if q not in W)},
        "by_id": by_id,
    }
    p = Path(args.ids).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["counts"]), "D_allocation", alloc, "->", p)


# ----------------------------------------------------------------------------- run
def prompt_rows(arm_dir: Path, dom: str, want: set) -> dict:
    rows = {}
    p = arm_dir / f"{dom}_small/prompt_rows.jsonl"
    if not p.exists():
        return rows
    with open(p) as fp:
        for line in fp:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["question_id"] in want:
                rows[r["question_id"]] = r
            del r
    return rows


async def call_one(client, model, row, args, sem):
    async with sem:
        r = await client.chat.completions.create(
            model=model, messages=row["messages"], max_tokens=args.max_tokens, timeout=args.timeout,
            temperature=0.6, top_p=0.95, extra_body={"top_k": 20})
    m = r.choices[0].message
    content = m.content if isinstance(m.content, str) else ""
    return {
        "question_id": row["question_id"],
        "reasoning": reasoning_of(m),
        "content": content,
        "boxed": extract_boxed_answer(content.strip()),
        "completion_tokens": r.usage.completion_tokens if r.usage else None,
        "prompt_tokens": r.usage.prompt_tokens if r.usage else None,
        "finish_reason": r.choices[0].finish_reason,
    }


async def run_async(args) -> None:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=os.environ["READER_BASE_URL"], api_key="EMPTY")
    model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
    sem = asyncio.Semaphore(args.concurrency)
    ids = load_ids(args.ids)
    src = E1 / "runs" / ARMS[args.arm]
    for dom in DOMS:
        want = {q for q, e in ids["by_id"].items() if args.arm in e["arms"] and e["domain"] == dom}
        if args.limit:
            want = set(sorted(want)[: args.limit])
        dst = think_dir(args.arm, dom)
        dst.mkdir(parents=True, exist_ok=True)
        out = dst / "thinking.jsonl"
        done = set()
        if out.exists():
            for line in open(out):
                try:
                    done.add(json.loads(line)["question_id"])
                except Exception:
                    pass
        rows = prompt_rows(src, dom, want - done)
        missing = sorted((want - done) - set(rows))
        fallback = {}
        if missing and args.arm != "b1":  # fusion reused B1's prompt for questions whose blocks did not change
            fallback = prompt_rows(E1 / "runs" / ARMS["b1"], dom, set(missing))
            rows.update(fallback)
        missing = sorted((want - done) - set(rows))
        print(f"[think] {args.arm} {dom}: {len(want)} wanted, {len(done)} done, {len(rows)} to run, "
              f"{len(fallback)} from b1 rows, {len(missing)} missing {missing}", flush=True)
        for q, r in rows.items():
            r["_src"] = "b1_small(fallback)" if q in fallback else ARMS[args.arm]
        tasks = [call_one(client, model, r, args, sem) for r in rows.values()]
        with open(out, "a") as fp:
            for fut in asyncio.as_completed(tasks):
                try:
                    o = await fut
                except Exception as exc:  # keep going; the question stays undone and is retried on resume
                    print(f"[think] request failed: {exc!r}", flush=True)
                    continue
                r = rows[o["question_id"]]
                o.update({"arm": args.arm, "source_arm": r["_src"], "domain": dom,
                          "eval_function": r["eval_function"], "answer_gold": r["answer_gold"],
                          "memory_context_was_truncated": r.get("memory_context_was_truncated"),
                          "memory_context_token_count": r.get("memory_context_token_count")})
                fp.write(json.dumps(o, ensure_ascii=False) + "\n"); fp.flush()
                print(f"[think] {o['question_id']} tokens={o['completion_tokens']} finish={o['finish_reason']} "
                      f"think_chars={len(o['reasoning'])} boxed={o['boxed'][:60]!r}", flush=True)
    await client.close()


# ----------------------------------------------------------------------------- score
def original_scores(arm_folder: str, dom: str, want: set) -> dict:
    out = {}
    p = E1 / f"runs/{arm_folder}/{dom}_small/per_question.jsonl"
    with open(p) as fp:
        for line in fp:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["question_id"] in want:
                out[r["question_id"]] = bool(r["score_bool"])
            del r
    return out


def fusion_original() -> dict:
    """e2_fusion_small/per_question.jsonl was written under NO_JUDGE (scores absent), so the fusion
    arm's original rule scores come from rule_compare's dump, as in the E2 report."""
    p = E1 / "e2work/reports/r4_rule_fusion_complete.jsonl"
    return {r["question_id"]: bool(r["arm_correct"]) for r in (json.loads(l) for l in open(p) if l.strip())}


def score(args) -> None:
    ids = load_ids(args.ids)
    fo = fusion_original()
    table = defaultdict(Counter)  # (arm, group) -> outcome counts
    for arm, folder in ARMS.items():
        for dom in DOMS:
            p = think_dir(arm, dom) / "thinking.jsonl"
            if not p.exists():
                continue
            want = {q for q, e in ids["by_id"].items() if arm in e["arms"] and e["domain"] == dom}
            orig = original_scores(folder, dom, want)
            if arm == "fusion":
                orig = {q: fo[q] for q in want if q in fo}
            rows = {}
            for line in open(p):
                if line.strip():
                    o = json.loads(line)
                    rows[o["question_id"]] = o  # last line wins
            sc = []
            for q, o in rows.items():
                ok = (not is_unknown(o["boxed"])) and score_to_bool(eval_from_spec(o["eval_function"], o["boxed"], o["answer_gold"]))
                e = ids["by_id"].get(q, {})
                rec = {"question_id": q, "arm": arm, "domain": dom, "question_type": e.get("question_type"),
                       "groups": e.get("groups"), "rerun_correct": bool(ok), "original_correct": orig.get(q),
                       "boxed": o["boxed"], "answer_gold": o["answer_gold"], "finish_reason": o["finish_reason"],
                       "completion_tokens": o["completion_tokens"], "cap_hit": o["finish_reason"] == "length"}
                sc.append(rec)
                for g in e.get("groups", []):
                    key = ("wrong_both" if not ok and orig.get(q) is False else
                           "wrong_orig_right_rerun" if ok and orig.get(q) is False else
                           "right_orig_wrong_rerun" if not ok and orig.get(q) else "right_both")
                    table[(arm, g)][key] += 1
                    table[(arm, g)]["n"] += 1
            with open(think_dir(arm, dom) / "scores.jsonl", "w") as fp:
                for rec in sc:
                    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_missing = len(want - set(rows))
            print(f"[score] {arm} {dom}: {len(sc)} scored, {n_missing} missing, "
                  f"rerun correct {sum(r['rerun_correct'] for r in sc)}, original correct {sum(bool(r['original_correct']) for r in sc)}")
    print("\nreproduction table (original run vs thinking rerun; rows = arm x group)")
    print(f"{'arm':7}{'group':9}{'n':>4}{'wrong both':>12}{'orig wrong, rerun right':>25}{'orig right, rerun wrong':>25}{'right both':>12}")
    for (arm, g), c in sorted(table.items()):
        print(f"{arm:7}{g:9}{c['n']:>4}{c['wrong_both']:>12}{c['wrong_orig_right_rerun']:>25}{c['right_orig_wrong_rerun']:>25}{c['right_both']:>12}")


# ----------------------------------------------------------------------------- export-c
def export_c(args) -> None:
    ids = load_ids(args.ids)
    fo = fusion_original()
    C = set(ids["groups"]["C_fixed"]) | set(ids["groups"]["C_broke"])
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ("question_id", "question_type", "eval_function", "question_text", "answer_gold", "response_parsed_boxed",
            "is_unknown", "score", "score_bool", "usage", "memory_context_was_truncated",
            "memory_context_token_count", "memory_context_original_token_count")
    n = 0
    with open(out, "w") as fp:
        for arm, folder in ARMS.items():
            for dom in DOMS:
                with open(E1 / f"runs/{folder}/{dom}_small/per_question.jsonl") as src:
                    for line in src:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        if r["question_id"] in C:
                            rec = {"arm": arm, "domain": dom, **{k: r.get(k) for k in keep},
                                   "group": "C_fixed" if r["question_id"] in ids["groups"]["C_fixed"] else "C_broke",
                                   "response_raw_tail": (r.get("response_raw") or "")[-600:]}
                            if arm == "fusion":  # per_question scores absent (NO_JUDGE); use the rule_compare dump
                                rec["rule_correct_r4"] = fo.get(r["question_id"])
                            fp.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
                        del r
    print(f"[export-c] {n} rows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["select", "run", "score", "export-c"])
    ap.add_argument("--ids", default=str(IDS_DEFAULT))
    ap.add_argument("--arm", choices=list(ARMS), default="b1")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(E1 / "e2work/reports/error_analysis_C_rows.jsonl"))
    a = ap.parse_args()
    if a.mode == "select":
        select(a)
    elif a.mode == "run":
        asyncio.run(run_async(a))
    elif a.mode == "score":
        score(a)
    else:
        export_c(a)


if __name__ == "__main__":
    main()
