#!/usr/bin/env python3
"""Derive an experiment arm from a finished baseline run without re-running retrieval.

Reads <src>/<domain>_small/prompt_rows.jsonl (the baseline's exact memory contexts) and writes
<dst>/<domain>_small/prompt_rows.jsonl for a new arm. The harness then reuses these rows
(see the cached-prompt branch in evaluation/harness.py), so only the reader runs again and
retrieval is byte-identical across arms.

Transforms (combinable):
  --fix-prompt     Replace the backspace (0x08) that the official system prompt contains where
                   "\\boxed{}" was intended (upstream issue #11). Arm B1.
  --premise-check  Before the reader, ask the controller (same Qwen3.5-9B endpoint) whether the
                   question's assumptions are SUPPORTED / CONTRADICTED / NOT_FOUND in the retrieved
                   text, and prepend a short note to the memory context when CONTRADICTED. Arm P.
                   The checker sees only the question and the retrieved context: no gold answer,
                   no question_type, no abstention label. Its wall time is added to
                   memory_query_duration_seconds so latency accounting includes it.
                   Verdicts are checkpointed to <dst>/<domain>_small/premise_check.jsonl.
  --check-only     With --premise-check: run the checker and write verdicts, but no prompt rows
                   (cheap gate: inspect recall / false-alarm before paying for a reader run).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BACKSPACE_BOXED = "\x08oxed{}"
FIXED_BOXED = "\\boxed{}"

CHECK_SYSTEM = (
    "You audit questions about a web environment against retrieved memory of that environment. "
    "You never answer the question itself."
)
CHECK_TEMPLATE = """Retrieved memory (text only; screenshots omitted):
<<<MEMORY
{memory}
MEMORY>>>

Question:
{question}

Task: list the factual assumptions this question makes about the environment, i.e. elements,
fields, options, links, columns, steps or behaviours that the question presupposes exist or happen.
For each assumption decide, using ONLY the retrieved memory above:
- SUPPORTED: the memory shows it.
- CONTRADICTED: the memory shows the relevant page, form or state, and the assumed thing is absent or different there.
- NOT_FOUND: the memory does not show the relevant page, form or state.

Reply with JSON only, no prose:
{{"assumptions": [{{"text": "...", "status": "SUPPORTED|CONTRADICTED|NOT_FOUND", "evidence": "short quote or description from memory"}}],
  "overall": "SUPPORTED|CONTRADICTED|NOT_FOUND"}}
"overall" is CONTRADICTED if any assumption is CONTRADICTED, else NOT_FOUND if any is NOT_FOUND, else SUPPORTED."""

NOTE_TEMPLATE = """## Premise check (automatic; may be wrong)
Before answering, note that the retrieved memory appears to contradict an assumption in the question:
- Assumption: {assumption}
- What the memory shows: {evidence}
Verify this against the evidence below. If you agree the question's premise is wrong, do not answer the
question as asked; instead explain inside \\boxed{{}} why the question is flawed.
"""


def log(msg: str) -> None:
    print(f"[derive_arm] {msg}", flush=True)


def fix_messages(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str) and BACKSPACE_BOXED in m["content"]:
            m["content"] = m["content"].replace(BACKSPACE_BOXED, FIXED_BOXED)
            n += 1
    return n


def memory_text(memory_context: list[dict], max_chars: int) -> str:
    text = "\n\n".join(x["value"] for x in memory_context if x["type"] == "text")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n...[{len(text) - max_chars} chars omitted]...\n" + text[-half:]


def parse_verdict(raw: str) -> dict:
    raw = raw or ""
    obj = None
    start = raw.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict):
        overall = str(obj.get("overall", "")).upper()
        assumptions = obj.get("assumptions") or []
    else:  # fall back to the overall field alone
        m = re.search(r'"overall"\s*:\s*"(SUPPORTED|CONTRADICTED|NOT_FOUND)"', raw, re.I)
        overall = m.group(1).upper() if m else ""
        assumptions = []
    if overall not in {"SUPPORTED", "CONTRADICTED", "NOT_FOUND"}:
        return {"overall": "PARSE_ERROR", "assumptions": assumptions, "raw": raw[:4000]}
    return {"overall": overall, "assumptions": assumptions}


def run_check(client, model: str, row: dict, args) -> dict:
    prompt = CHECK_TEMPLATE.format(memory=memory_text(row["memory_context"], args.check_max_chars),
                                   question=row["question_text"])
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=args.check_max_tokens, temperature=args.check_temperature, top_p=0.95,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": args.check_thinking}},
        timeout=600,
    )
    dt = time.perf_counter() - t0
    raw = resp.choices[0].message.content or ""
    v = parse_verdict(raw)
    v.update({"question_id": row["question_id"], "check_seconds": dt,
              "completion_tokens": getattr(resp.usage, "completion_tokens", None)})
    return v


# QUOTED / quoted_terms: identical to the frozen rule in e1/oneoff/strict_rule.py (committed 5894287
# before enterprise was examined); now shared via e1/common/anchors.py.
from e1.common.anchors import QUOTED, quoted_terms  # noqa: E402,F401


NOTE_NF_TEMPLATE = """## Premise check (automatic; may be wrong)
The retrieved memory shows no evidence for an element the question takes for granted:
- Assumed element: {assumption}
- What the memory shows: {evidence}
Check the evidence below. If the relevant page or form is covered and the element is absent, the question's
premise is probably wrong: explain inside \\boxed{{}} why the question is flawed. If the memory simply does not
cover that page, answer from the evidence you have.
"""


def note_from_verdict(v: dict, question: str = "", rule: str = "any") -> str | None:
    if rule == "nfq":  # frozen on the web dev split, commit after 5894287 (see log 02:25)
        if v.get("overall") not in {"CONTRADICTED", "NOT_FOUND"}:
            return None
        terms = quoted_terms(question)
        hits = [a for a in v.get("assumptions", [])
                if str(a.get("status", "")).upper() in {"CONTRADICTED", "NOT_FOUND"}
                and any(t in str(a.get("text", "")).lower() for t in terms)]
        if not hits:
            return None
        a = sorted(hits, key=lambda x: str(x.get("status", "")).upper() != "CONTRADICTED")[0]
        tmpl = NOTE_TEMPLATE if str(a.get("status", "")).upper() == "CONTRADICTED" else NOTE_NF_TEMPLATE
        return tmpl.format(assumption=str(a.get("text", "(unspecified)"))[:400],
                           evidence=str(a.get("evidence", "(unspecified)"))[:600])
    if v.get("overall") != "CONTRADICTED":
        return None
    contra = [a for a in v.get("assumptions", []) if str(a.get("status", "")).upper() == "CONTRADICTED"]
    if rule == "strict":
        terms = quoted_terms(question)
        contra = [a for a in contra if any(t in str(a.get("text", "")).lower() for t in terms)]
        if not contra:
            return None
    a = contra[0] if contra else {}
    return NOTE_TEMPLATE.format(assumption=str(a.get("text", "(unspecified)"))[:400],
                                evidence=str(a.get("evidence", "(unspecified)"))[:600])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="baseline run root, e.g. ~/e1/runs/agentrunbook_r_small")
    ap.add_argument("--dst", required=True, help="new arm run root")
    ap.add_argument("--domains", default="web,enterprise")
    ap.add_argument("--fix-prompt", action="store_true")
    ap.add_argument("--premise-check", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--check-thinking", action="store_true", help="enable Qwen thinking for the checker (slower)")
    ap.add_argument("--check-max-tokens", type=int, default=1536)
    ap.add_argument("--check-max-chars", type=int, default=400_000)
    ap.add_argument("--check-temperature", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--note-rule", choices=["any", "strict", "nfq"], default="any")
    ap.add_argument("--ids-from-capped", action="store_true",
                    help="keep only questions whose SOURCE reader output hit the 20k completion cap")
    ap.add_argument("--only-noted", action="store_true",
                    help="write prompt rows only for questions that receive a note (others are identical to B1)")
    ap.add_argument("--latency-sample", type=int, default=0,
                    help="after the concurrent run, time N checker calls one at a time (clean per-query latency)")
    ap.add_argument("--memory-context-max-tokens", type=int, default=200000)
    args = ap.parse_args()

    client = model = None
    if args.premise_check:
        from openai import OpenAI
        client = OpenAI(base_url=os.environ.get("READER_BASE_URL", "http://localhost:8023/v1"),
                        api_key=os.environ.get("READER_API_KEY", "EMPTY"))
        model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
    if not args.check_only:
        from evaluation.harness import build_messages, truncate_memory_context  # needs GPU node (transformers)

    for dom in args.domains.split(","):
        src = Path(args.src).expanduser() / f"{dom}_small"
        dst = Path(args.dst).expanduser() / f"{dom}_small"
        dst.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(l) for l in open(src / "prompt_rows.jsonl") if l.strip()]
        if args.limit:
            rows = rows[: args.limit]
        if args.ids_from_capped:
            capped = set()
            for l in open(src / "reader_outputs_cache.jsonl"):
                c = json.loads(l)
                if (c["output"].get("usage") or {}).get("completion_tokens", 0) >= 20000:
                    capped.add(c["question_id"])
            rows = [r for r in rows if r["question_id"] in capped]
        log(f"{dom}: {len(rows)} rows from {src}")

        verdicts: dict[str, dict] = {}
        if args.premise_check:
            vpath = dst / "premise_check.jsonl"
            if vpath.exists():
                for l in open(vpath):
                    try:
                        v = json.loads(l); verdicts[v["question_id"]] = v
                    except (json.JSONDecodeError, KeyError):
                        pass
            todo = [r for r in rows if r["question_id"] not in verdicts]
            log(f"{dom}: premise check {len(verdicts)} cached, {len(todo)} to run")
            with open(vpath, "a") as vfp, ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(run_check, client, model, r, args): r["question_id"] for r in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    try:
                        v = f.result()
                    except Exception as exc:  # keep going; record the failure
                        v = {"question_id": futs[f], "overall": "CALL_ERROR", "error": str(exc)[:300],
                             "assumptions": [], "check_seconds": None}
                    verdicts[v["question_id"]] = v
                    vfp.write(json.dumps(v) + "\n"); vfp.flush()
                    if i % 25 == 0:
                        log(f"{dom}: checked {i}/{len(todo)}")
        if args.premise_check and args.latency_sample:
            import random
            sample = random.Random(0).sample(rows, min(args.latency_sample, len(rows)))
            with open(dst / "premise_check_latency.jsonl", "w") as lfp:
                for r in sample:
                    v = run_check(client, model, r, args)
                    lfp.write(json.dumps({"question_id": r["question_id"], "check_seconds": v["check_seconds"],
                                          "completion_tokens": v.get("completion_tokens"),
                                          "context_tokens": r.get("memory_context_token_count")}) + "\n")
            log(f"{dom}: sequential latency sample of {len(sample)} written")
        if args.check_only:
            continue

        n_fixed = n_noted = 0
        out_path = dst / "prompt_rows.jsonl"
        with open(out_path, "w") as fp:
            for r in rows:
                mc = list(r["memory_context"])
                note = (note_from_verdict(verdicts.get(r["question_id"], {}), r["question_text"], args.note_rule)
                        if args.premise_check else None)
                if args.only_noted and note is None:
                    continue
                if note is not None:
                    mc = [{"type": "text", "value": note}] + mc
                    mc, orig_tok, trunc_tok = truncate_memory_context(
                        mc, max_tokens=args.memory_context_max_tokens, question_id=r["question_id"])
                    r["memory_context_token_count"] = trunc_tok
                    n_noted += 1
                system_prompt = next(m["content"] for m in r["messages"] if m["role"] == "system")
                if args.fix_prompt and BACKSPACE_BOXED in system_prompt:
                    system_prompt = system_prompt.replace(BACKSPACE_BOXED, FIXED_BOXED); n_fixed += 1
                if note is not None or args.fix_prompt:
                    r["messages"], r["prompt_messages"] = build_messages(
                        system_prompt=system_prompt, question_text=r["question_text"],
                        image_path=r["question_image"], memory_context=mc)
                    r["memory_context"] = mc
                if args.premise_check:
                    v = verdicts.get(r["question_id"], {})
                    r["memory_query_duration_seconds"] = (r.get("memory_query_duration_seconds") or 0.0) + (v.get("check_seconds") or 0.0)
                    r["e1_premise_verdict"] = v.get("overall")
                    r["e1_premise_note_added"] = note is not None
                fp.write(json.dumps(r, ensure_ascii=True) + "\n")
        if args.only_noted or args.ids_from_capped:
            (dst / "noted_ids.txt").write_text(",".join(
                json.loads(l)["question_id"] for l in open(out_path) if l.strip()))
        log(f"{dom}: wrote {out_path} (system prompt fixed in {n_fixed}, premise note added to {n_noted})")


if __name__ == "__main__":
    main()
