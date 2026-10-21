#!/usr/bin/env python3
"""R4 controls (post hoc): for each rule-scored question, cross the reader outcome change
(fixed / broke / same, arm vs B1) with
  - layer-4 coverage change: answer-phrase coverage of the arm's raw blocks vs B1's blocks
    (gain / loss / same-covered / same-uncovered; B1 blocks known only for complete questions)
  - truncation: was the arm's reader context truncated at 200k, and how many raw blocks survived
    (retrieved vs left in context); same for B1 (blocks left only)
and print the counts per question type, plus truncation counts by type for arm vs B1.
Input: --dump from rule_compare.py (per-question rule scores).
Usage: python -I e1/analysis/r4_cross.py --dump ~/e1/e2work/reports/r4_rule_fusion.jsonl --arm e2_fusion_small --ret e2_fusion
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.analysis.retrieval_eval import Content, label_question  # noqa: E402
from e1.common.io import iter_jsonl  # noqa: E402
from e1.pipeline.retrieve import meta  # noqa: E402

E1 = Path("~/e1").expanduser()


def ctx_info(path: Path, want: set) -> dict:
    out = {}
    with open(path) as fp:
        for line in fp:
            r = json.loads(line)
            if r["question_id"] in want:
                n = sum(1 for x in r["memory_context"] if x.get("type") == "text" and x["value"].startswith("### Raw state result "))
                out[r["question_id"]] = (bool(r.get("memory_context_was_truncated")), n)
            del r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--arm", default="e2_fusion_small")
    ap.add_argument("--ret", default="e2_fusion")
    a = ap.parse_args()
    D = {r["question_id"]: r for r in iter_jsonl(Path(a.dump).expanduser())}
    rows = []
    for dom in ["web", "enterprise"]:
        ids = {q for q, r in D.items() if r["domain"] == dom}
        C, M = Content(dom), meta(dom)
        Q = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl")}
        R = {r["question_id"]: r for r in iter_jsonl(E1 / f"e2work/retrieved/{a.ret}/{dom}.jsonl")}
        C.load({b["entry_id"] for q in ids for b in R[q]["raw_blocks"] + Q[q]["baseline_raw_blocks"]})
        ta = ctx_info(E1 / f"runs/{a.arm}/{dom}_small/prompt_rows.jsonl", ids)
        tb = ctx_info(E1 / f"runs/b1_small/{dom}_small/prompt_rows.jsonl", ids)
        for q in ids:
            d = D[q]
            outcome = "fixed" if d["arm_correct"] and not d["ref_correct"] else "broke" if d["ref_correct"] and not d["arm_correct"] else "same"
            lx = label_question(M[q], [b["entry_id"] for b in R[q]["raw_blocks"]], C)
            if Q[q]["context_truncated"]:
                cov = "n/a (B1 blocks unknown)"
            elif "ans" not in lx:
                cov = "n/a (no phrase answer)"
            else:
                lb = label_question(M[q], [b["entry_id"] for b in Q[q]["baseline_raw_blocks"]], C)
                cov = "gain" if lx["ans"] and not lb["ans"] else "loss" if lb["ans"] and not lx["ans"] else \
                    "same-covered" if lx["ans"] else "same-uncovered"
            tra, nleft = ta[q]
            nret = len(R[q]["raw_blocks"])
            trunc = "arm not truncated" if not tra else ("arm truncated, all raw cut" if nleft == 0 else
                                                          "arm truncated, part raw cut" if nleft < nret else "arm truncated, raw intact")
            rows.append({**d, "outcome": outcome, "coverage_change": cov, "arm_trunc": trunc, "b1_truncated": tb[q][0],
                         "arm_raw_left": nleft, "arm_raw_retrieved": nret, "b1_raw_left": tb[q][1]})
    out = Path(a.dump).with_name(Path(a.dump).stem + "_cross.jsonl")
    with open(out, "w") as fp:
        for r in rows:
            fp.write(json.dumps(r) + "\n")
    for t in ["static-environment", "dynamic-environment", "procedure"]:
        S = [r for r in rows if r["question_type"] == t]
        print(f"\n== {t} (n={len(S)}), outcome x coverage change (complete questions only have B1 blocks)")
        c = Counter((r["outcome"], r["coverage_change"]) for r in S)
        for k, v in sorted(c.items()):
            print(f"   {k[0]:6s} {k[1]:26s} {v}")
        print("   outcome x arm truncation:")
        c = Counter((r["outcome"], r["arm_trunc"]) for r in S)
        for k, v in sorted(c.items()):
            print(f"   {k[0]:6s} {k[1]:30s} {v}")
        print(f"   truncated contexts: arm {sum(r['arm_trunc'] != 'arm not truncated' for r in S)}/{len(S)}, "
              f"B1 {sum(r['b1_truncated'] for r in S)}/{len(S)}")
    print("\nfixed/broke questions with details:")
    for r in sorted(rows, key=lambda r: (r["question_type"], r["outcome"])):
        if r["outcome"] != "same" and r["question_type"] in ("static-environment", "dynamic-environment"):
            print(f"   {r['question_type'][:7]} {r['outcome']:5s} {r['question_id']} {r['domain']:10s} cov={r['coverage_change']:22s} "
                  f"{r['arm_trunc']:28s} raw left {r['arm_raw_left']}/{r['arm_raw_retrieved']}  B1 truncated={r['b1_truncated']} B1 raw left {r['b1_raw_left']}")


if __name__ == "__main__":
    main()
