#!/usr/bin/env python3
"""Arm vs B1 on RULE-SCORED questions only (no LLM judge), paired exact McNemar, by question type.

Used while the GPT-5.2 judge is unavailable (OpenAI credits exhausted, 2026-10-22 05:05).
Rule-scored = eval_function is not llm_abstention_checker / llm_gotchas_checker, i.e. normal
static / dynamic / procedure questions with phrase-set or multiple-choice answers. B1's scores on
these questions are also rule-based (the harness only calls the judge for the two LLM checkers),
so the comparison is like-for-like. Arm outputs are read from reader_outputs_cache.jsonl (last valid
line per question, as the harness does; corrupted lines are skipped) and scored with the official
qa_eval_metrics (UNKNOWN counts as wrong, as in the harness).
Optional --subset complete|regenerated|all and --ids-file restrict the questions.
Usage: python3 e1/analysis/rule_compare.py e2_fusion_small [--subset complete]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.common.stats import mcnemar_exact  # noqa: E402

E1 = Path("~/e1").expanduser()
LLM = {"llm_abstention_checker", "llm_gotchas_checker"}


def arm_outputs(arm: str, dom: str) -> dict:
    out = {}
    p = E1 / f"runs/{arm}/{dom}_small/reader_outputs_cache.jsonl"
    if not p.exists():
        return out
    with open(p, encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(c.get("output"), dict):
                out[c["question_id"]] = c["output"]
    return out


def main() -> None:
    from evaluation.qa_eval_metrics import eval_from_spec, eval_name, score_to_bool
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--ref", default="b1_small")
    ap.add_argument("--subset", default="all", choices=["all", "complete", "regenerated"])
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--dump", default=None, help="write per-question rule scores (jsonl)")
    a = ap.parse_args()
    keep = {x for x in Path(a.ids_file).expanduser().read_text().split(",") if x} if a.ids_file else None
    groups = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, ref, arm, fixed, broke
    rows = []
    missing = 0
    for dom in ["web", "enterprise"]:
        M = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/meta/{dom}.jsonl")}
        Q = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl")}
        ref = {}
        for r in iter_jsonl(E1 / f"runs/{a.ref}/{dom}_small/per_question.jsonl"):
            ref[r["question_id"]] = bool(r["score_bool"]); del r
        O = arm_outputs(a.arm, dom)
        for q, m in M.items():
            if eval_name(m["eval_function"]) in LLM:
                continue
            regen = Q[q]["context_truncated"] or Q[q]["missing_detected"]
            if a.subset == "complete" and regen or a.subset == "regenerated" and not regen:
                continue
            if keep is not None and q not in keep:
                continue
            if q not in O:
                missing += 1
                continue
            o = O[q]
            ok = (not o["is_unknown"]) and score_to_bool(eval_from_spec(m["eval_function"], o["response_parsed_boxed"], m["answer_gold"]))
            r0 = ref[q]
            for g in [("ALL", "ALL"), (dom, "ALL"), ("ALL", m["question_type"]), (dom, m["question_type"])]:
                x = groups[g]
                x[0] += 1; x[1] += r0; x[2] += ok; x[3] += ok and not r0; x[4] += r0 and not ok
            rows.append({"question_id": q, "domain": dom, "question_type": m["question_type"], "ref_correct": r0,
                         "arm_correct": bool(ok), "regenerated": bool(regen)})
    print(f"{a.arm} vs {a.ref}, rule-scored normal questions, subset={a.subset}; arm outputs missing for {missing}")
    for (dom, t), (n, k0, k1, fx, br) in sorted(groups.items()):
        print(f"  {dom:10s} {t:22s} n={n:3d}  {a.ref} {k0:3d} ({k0/n:.3f})  {a.arm} {k1:3d} ({k1/n:.3f})  d={(k1-k0)/n:+.3f}  "
              f"fixed {fx} broke {br}  McNemar p={mcnemar_exact(br, fx):.3f}")
    if a.dump:
        with open(a.dump, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
