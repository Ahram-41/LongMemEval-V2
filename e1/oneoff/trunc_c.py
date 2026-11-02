#!/usr/bin/env python3
"""C (pre-registered in CP5101 experiments/trunc-plan-2026-11-02.md): within-block truncation.

The official truncate_memory_context drops whole items from the tail once the memory context is over
200k tokens, so whole raw-state blocks go first. C keeps every item and cuts only the AXTree body of
each state: one cap c for all states (water-filling), head kept, official '...[truncated N chars]...'
marker. Block headers, state headers (State / URL / Action), notes, events and screenshots untouched.

Token counts: the official counter (AutoProcessor) cannot load on CPU nodes, so the C context size is
estimated as the official original count (A0 prompt rows) minus the text-tokenizer count of what is cut
(Qwen3.5-9B tokenizer.json), with a 3k margin; the official check is redone on the GPU node when the
prompt rows are built.

Variants: head (C, primary) | mid (official _truncate_middle per state) | chrome (drop pool-chrome lines, then head)

  build : contexts for the 107 regenerated questions -> ~/e1/v2work/trunc/<variant>/<dom>_contexts.jsonl
  eval  : operation checks + answer-phrase proxy + frozen evidence rules, C vs A0 (official truncation)
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_keys  # noqa: E402
from e1.common.io import iter_jsonl  # noqa: E402

E1 = step_keys.E1
OUT = step_keys.V2 / "trunc"
MAX_TOKENS = 200_000
MARGIN = 3_000
A0_RUNS = ["e2_ordA0_small", "e2_ordA0rest_small"]
SPLIT_STATES = re.compile(r"\n\n(?=State \d+ \(step \d+\)\n- URL: )")
BODY_SEP = "Relevant state slices\n"
AX_SEP = "\n- AXTree:\n"


def tokenizer():
    import tokenizers
    f = glob.glob(str(E1 / "hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/*/tokenizer.json"))
    return tokenizers.Tokenizer.from_file(f[0])


def truncate_middle(text: str, max_chars: int) -> str:
    """Copy of memory_modules.agentrunbook_r._truncate_middle (importing the module needs the official stack)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 64:
        return text[:max_chars]
    omitted = len(text) - max_chars
    marker = f"\n...[truncated {omitted} chars]...\n"
    if len(marker) >= max_chars:
        return text[:max_chars]
    head = (max_chars - len(marker)) // 2
    tail = max_chars - len(marker) - head
    return text[:head] + marker + text[-tail:]


def parse_block(value: str):
    """'### Raw state result' text -> (head, [(state_header, axtree)], tail); exact round trip."""
    head, body = value.split(BODY_SEP, 1)
    tail = "\n" if body.endswith("\n") else ""
    body = body[:-1] if tail else body
    states = []
    for s in SPLIT_STATES.split(body):
        h, ax = s.split(AX_SEP, 1)
        states.append((h, ax))
    return head + BODY_SEP, states, tail


def render_block(head, states, tail) -> str:
    return head + "\n\n".join(h + AX_SEP + ax for h, ax in states) + tail


def is_raw(item) -> bool:
    return item.get("type") == "text" and item["value"].startswith("### Raw state result ")


def a0_rows(dom: str, runs=None) -> dict:
    """Reference prompt rows (official truncation): A0 by default, or e.g. rr_small for C2."""
    out = {}
    for d in runs or A0_RUNS:
        for r in iter_jsonl(E1 / f"runs/{d}/{dom}_small/prompt_rows.jsonl"):
            out[r["question_id"]] = {"mc": r["memory_context"], "orig": int(r["memory_context_original_token_count"]),
                                     "trunc": str(r["memory_context_was_truncated"]) == "True", "row": r}
    return out


def chrome_lines(dom: str) -> set[str]:
    df, n = Counter(), 0
    for e in step_keys.pool(dom):
        n += 1
        df.update({ln.strip() for ln in e["slice_axtree_text"].splitlines() if ln.strip()})
    return {k for k, c in df.items() if c >= step_keys.CHROME_DF * n}


def waterfill(lengths: list[int], budget: int) -> int:
    """Largest c with sum(min(n, c)) <= budget."""
    if sum(lengths) <= budget:
        return max(lengths, default=0)
    lo, hi = 0, max(lengths)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in lengths) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def cut_context(mc: list, orig_tokens: int, variant: str, tok, chrome: set[str] | None):
    idx = [i for i, x in enumerate(mc) if is_raw(x)]
    blocks = {i: parse_block(mc[i]["value"]) for i in idx}
    for i in idx:  # parser round trip must be exact
        assert render_block(*blocks[i]) == mc[i]["value"], f"parse round trip failed at item {i}"
    axs = [(i, k, ax) for i in idx for k, (_, ax) in enumerate(blocks[i][1])]
    pre_removed = 0
    if variant == "chrome":
        new = []
        for i, k, ax in axs:
            kept = "\n".join(ln for ln in ax.split("\n") if ln.strip() not in chrome)
            pre_removed += len(tok.encode(ax).ids) - len(tok.encode(kept).ids)
            new.append((i, k, kept))
        axs = new
    enc = [tok.encode(ax) for _, _, ax in axs]
    n = [len(e.ids) for e in enc]
    excess = orig_tokens - pre_removed - (MAX_TOKENS - MARGIN)
    stats = {"states": len(axs), "excess": excess, "iters": 0}
    out = [dict(x) for x in mc]
    for it in range(6):
        stats["iters"] = it + 1
        c = waterfill(n, sum(n) - max(excess, 0))
        removed = pre_removed
        new_states = {i: list(blocks[i][1]) for i in idx}
        for (i, k, ax), e, nk in zip(axs, enc, n):
            if nk <= c:
                kept = ax
            elif variant == "mid":
                kept = truncate_middle(ax, max(1, int(len(ax) * c / nk)))
            else:
                end = e.offsets[c - 1][1] if c > 0 else 0
                kept = ax[:end] + f"\n...[truncated {len(ax) - end} chars]...\n"
            removed += nk - len(tok.encode(kept).ids)
            new_states[i][k] = (new_states[i][k][0], kept)
        est = orig_tokens - removed
        if est <= MAX_TOKENS - MARGIN or excess <= 0:
            break
        excess += est - (MAX_TOKENS - MARGIN) + 200
    for i in idx:
        out[i]["value"] = render_block(blocks[i][0], new_states[i], blocks[i][2])
    stats.update({"cap_tokens": c, "est_tokens": est, "orig_tokens": orig_tokens})
    return out, stats


def do_build(variants: list[str], runs, contexts: Path, prefix: str) -> None:
    tok = tokenizer()
    for dom in step_keys.DOMS:
        A0 = a0_rows(dom, runs)
        ctx = {r["question_id"]: r["memory_context"] for r in iter_jsonl(contexts / f"{dom}_contexts.jsonl")
               if r["question_id"] in A0}
        bad = [q for q in A0 if A0[q]["mc"] != ctx[q][:len(A0[q]["mc"])]]
        print(f"[check] {dom}: A0 rows {len(A0)}; A0 context is a prefix of the e2_v context for {len(A0) - len(bad)}", flush=True)
        assert not bad, bad[:5]
        chrome = chrome_lines(dom) if "chrome" in variants else None
        for v in variants:
            d = OUT / (prefix + v)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{dom}_truncated_ids.txt").write_text(",".join(sorted(q for q in A0 if A0[q]["trunc"])))
            with open(d / f"{dom}_contexts.jsonl", "w") as fp:
                for q in sorted(A0):
                    if A0[q]["trunc"]:
                        mc, st = cut_context(ctx[q], A0[q]["orig"], v, tok, chrome)
                    else:
                        mc, st = ctx[q], {"untouched": True}
                    fp.write(json.dumps({"question_id": q, "memory_context": mc, "stats": st}) + "\n")
            print(f"[build] {dom} {v}: {len(A0)} contexts", flush=True)


def text_of(mc: list) -> str:
    return "\n".join(x["value"] for x in mc if x.get("type") == "text")


def raw_states(mc: list, pool_idx: dict) -> list[tuple[dict, list[str]]]:
    """(pool entry, lower-cased state AXTree texts as rendered) per raw block in the context."""
    out = []
    for x in mc:
        if not is_raw(x):
            continue
        traj = re.search(r"^- Trajectory: (\S+)$", x["value"], re.M).group(1)
        center = int(re.search(r"^- Center state index: (\d+)$", x["value"], re.M).group(1))
        e = pool_idx.get(f"{traj}:raw_state:{center:04d}")
        _, states, _ = parse_block(x["value"])
        out.append((e, [ax.lower() for _, ax in states]))
    return out


def do_eval(variants: list[str], evidence: Path, out: Path, runs, contexts: Path, prefix: str) -> None:
    from e1.analysis.retrieval_eval import _official_metrics, answer_phrases
    from e1.pipeline.retrieve import meta
    normalize_phrase = _official_metrics()[0]
    rules = json.loads(evidence.read_text())["questions"]
    scored = {json.loads(l)["question_id"] for l in open(E1 / "e2work/reports/iv_scores.jsonl")}
    res = {"variants": variants, "ops": {}, "phrase": {}, "evidence": {}, "per_q": {}}
    for dom in step_keys.DOMS:
        A0, M = a0_rows(dom, runs), meta(dom)
        ctx0 = {r["question_id"]: r["memory_context"] for r in iter_jsonl(contexts / f"{dom}_contexts.jsonl") if r["question_id"] in A0}
        C = {v: {r["question_id"]: r for r in iter_jsonl(OUT / (prefix + v) / f"{dom}_contexts.jsonl")} for v in variants}
        R = {q8: r for q8, r in rules.items() if r["domain"] == dom}
        need = {q for q in A0 if A0[q]["trunc"] and any(q.startswith(q8) for q8 in R)}
        pool_idx = {e["entry_id"]: {"slice_urls": e["slice_urls"], "goal": e["goal"]} for e in step_keys.pool(dom)} if need else {}
        trunc = sorted(q for q in A0 if A0[q]["trunc"])
        for v in variants:
            ops = {"truncated": len(trunc), "over_est": 0, "blocks_lost": 0, "other_items_changed": 0, "untouched_differ": 0}
            for q in A0:
                c, mc0 = C[v][q]["memory_context"], ctx0[q]
                if not A0[q]["trunc"]:
                    ops["untouched_differ"] += c != mc0
                    continue
                ops["over_est"] += C[v][q]["stats"]["est_tokens"] > MAX_TOKENS - MARGIN
                ops["blocks_lost"] += sum(map(is_raw, c)) != sum(map(is_raw, mc0)) or len(c) != len(mc0)
                ops["other_items_changed"] += any(a != b for a, b in zip(c, mc0) if not is_raw(b))
            res["ops"].setdefault(v, {})[dom] = ops
        # answer-phrase proxy on rule-scored truncated questions
        for v in variants:
            g = l = el = 0
            for q in trunc:
                if q not in scored:
                    continue
                ph = answer_phrases(M[q])
                if ph is None:
                    continue
                el += 1
                t0 = " " + normalize_phrase(text_of(A0[q]["mc"])) + " "
                t1 = " " + normalize_phrase(text_of(C[v][q]["memory_context"])) + " "
                a, b = all(f" {p} " in t0 for p in ph), all(f" {p} " in t1 for p in ph)
                g += b and not a; l += a and not b
                res["per_q"].setdefault(q, {})[v] = {"a0": a, "c": b}
            res["phrase"].setdefault(v, {})[dom] = {"eligible": el, "gain": g, "loss": l}
        # frozen evidence rules
        for q in sorted(need):
            q8 = next(x for x in R if q.startswith(x))
            row = {}
            for name, mc in [("A0", A0[q]["mc"])] + [(v, C[v][q]["memory_context"]) for v in variants]:
                bl = raw_states(mc, pool_idx)
                row[name] = {g: any(e is not None and step_keys.matches(rule, e, st) for e, st in bl)
                             for g, rule in R[q8]["groups"].items()}
            res["evidence"][q8] = row
        res.setdefault("stats", {})[dom] = {v: {"median_cap_tokens": sorted(C[v][q]["stats"]["cap_tokens"] for q in trunc)[len(trunc) // 2]}
                                            for v in variants}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("ops", "phrase", "evidence", "stats")}, indent=1))


def do_rows_check(rows_dir: Path, untruncated: bool, same_as) -> None:
    """GPU-node check of built prompt rows: no further official truncation (C), or identical contexts (A0 rebuild)."""
    bad = 0
    for dom in step_keys.DOMS:
        ref = a0_rows(dom, same_as) if same_as else None
        n = 0
        for r in iter_jsonl(rows_dir / f"{dom}_small/prompt_rows.jsonl"):
            n += 1
            q = r["question_id"]
            if untruncated and str(r["memory_context_was_truncated"]) == "True":
                bad += 1; print(f"[rows-check] {dom} {q}: officially truncated again "
                                f"({r['memory_context_original_token_count']} tokens)")
            if ref is not None and r["memory_context"] != ref[q]["mc"]:
                bad += 1; print(f"[rows-check] {dom} {q}: context differs from the reference rows")
        print(f"[rows-check] {rows_dir.name} {dom}: {n} rows", flush=True)
        if n == 0:  # an empty id list once passed this check vacuously
            bad += 1; print(f"[rows-check] {dom}: no rows")
    print(f"[rows-check] {rows_dir.name}: {bad} problems", flush=True)
    sys.exit(1 if bad else 0)


def do_score(out: Path) -> None:
    """Pre-registered C1 / C2 readout: strict + forced wrap-up, 2 samples per arm (trunc-reader-plan §2)."""
    import collections
    from reader_interventions import correct, correct_strict
    def arm(v):
        res = {}
        for dom in step_keys.DOMS:
            p = E1 / f"runs/iv_{v}_small/{dom}_small/outputs.jsonl"
            if not p.exists():
                continue
            for o in iter_jsonl(p):
                ok = bool(correct(o["wrap2_boxed"], o)) if o["cap_no_answer"] else correct_strict(o["content"], o)
                res[o["question_id"]] = (ok, o["cap_no_answer"])
        return res
    F = collections.defaultdict(dict)
    for r in iter_jsonl(E1 / "e2work/reports/iv_scores.jsonl"):
        F[r["question_id"]][r["variant"]] = str(r["forced"]) == "True"
    stab = {q: "stable-wrong" if not any(v.values()) else "stable-right" if all(v.values()) else "unstable" for q, v in F.items()}
    report = {}
    for name, ctl, trt in [("C1", "c1a0", "c1c"), ("C2", "c2rr", "c2rc")]:
        A = [arm(f"{ctl}_s1"), arm(f"{ctl}_s2")]
        B = [arm(f"{trt}_s1"), arm(f"{trt}_s2")]
        qs = sorted(set.intersection(*[set(x) for x in A + B]))
        a = {q: sum(x[q][0] for x in A) for q in qs}
        b = {q: sum(x[q][0] for x in B) for q in qs}
        fix = [q for q in qs if a[q] == 0 and b[q] == 2]
        brk = [q for q in qs if a[q] == 2 and b[q] == 0]
        net = sum(b.values()) - sum(a.values())
        verdict = ("adopt" if len(fix) >= 3 and len(brk) <= 1 and net >= 4 else
                   "reject" if len(fix) <= len(brk) or net <= 0 else "inconclusive")
        by = collections.Counter((stab.get(q, "?"), b[q] - a[q]) for q in qs)
        report[name] = {"n": len(qs), "control_correct": sum(a.values()), "treatment_correct": sum(b.values()), "net": net,
                        "stable_fixed": fix, "stable_broken": brk, "verdict": verdict,
                        "caps": {"control": sum(x[q][1] for x in A for q in qs), "treatment": sum(x[q][1] for x in B for q in qs)},
                        "by_b1_stability": {f"{k[0]} {k[1]:+d}": c for k, c in sorted(by.items())},
                        "per_q": {q: [a[q], b[q], stab.get(q, "?")] for q in qs}}
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {x: v[x] for x in v if x != "per_q"} for k, v in report.items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "eval", "rows-check", "score"])
    ap.add_argument("--ref-runs", default=",".join(A0_RUNS), help="run dirs whose prompt rows are the official-truncation reference")
    ap.add_argument("--contexts", default=str(E1 / "e2work/contexts/e2_v"), help="pre-truncation contexts of those rows")
    ap.add_argument("--prefix", default="", help="output dir prefix under ~/e1/v2work/trunc (e.g. rr_)")
    ap.add_argument("--rows", default=None)
    ap.add_argument("--untruncated", action="store_true")
    ap.add_argument("--same-as", default=None)
    ap.add_argument("--variants", default="head,mid,chrome")
    ap.add_argument("--evidence", default=str(step_keys.V2 / "evidence.json"))
    ap.add_argument("--out", default=str(step_keys.V2 / "reports/trunc_c.json"))
    a = ap.parse_args()
    runs, ctx = a.ref_runs.split(","), Path(a.contexts).expanduser()
    if a.stage == "build":
        do_build(a.variants.split(","), runs, ctx, a.prefix)
    elif a.stage == "eval":
        do_eval(a.variants.split(","), Path(a.evidence).expanduser(), Path(a.out).expanduser(), runs, ctx, a.prefix)
    elif a.stage == "score":
        do_score(step_keys.V2 / "reports/trunc_reader.json")
    else:
        do_rows_check(Path(a.rows).expanduser(), a.untruncated, a.same_as.split(",") if a.same_as else None)


if __name__ == "__main__":
    main()
