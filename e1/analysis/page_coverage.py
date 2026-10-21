#!/usr/bin/env python3
"""Did retrieval bring back the page a trap (false-premise) question is about?

No official evidence map is published for LME-V2, so this uses a checkable proxy:
anchors = the UI terms the question names in quotes ("...", `...`, '...', “...”).
For each retrieved raw-state block (one "Raw state result" section of the memory context),
check which anchors appear in it.
  full    : some single block contains ALL anchors   -> the named page/form was retrieved
  partial : anchors appear, but never all in one block
  none    : no anchor appears in any block
  no-anchor : the question names no quoted term (proxy not applicable)
Cross-tabs the label with correctness for the given arms.
Usage: python e1/analysis/page_coverage.py ~/e1/runs agentrunbook_r_small b1_small pa_merged_small
The first arm supplies the retrieved context (identical across arms by construction).
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.anchors import coverage_label as label  # noqa: E402
from e1.common.io import DOMAINS, iter_jsonl  # noqa: E402


def main():
    root = Path(sys.argv[1]).expanduser()
    ref, arms = sys.argv[2], sys.argv[3:]
    lab, is_abs, qtype = {}, {}, {}
    for d in DOMAINS:
        for r in iter_jsonl(root / ref / d / "per_question.jsonl"):
            lab[r["question_id"]] = label(r["question_text"], r["memory_context"])
            is_abs[r["question_id"]] = r["is_abstention_problem"]
            qtype[r["question_id"]] = r["question_type"]
            del r
    score = defaultdict(dict)
    for a in [ref] + arms:
        for d in DOMAINS:
            p = root / a / d / "per_question.jsonl"
            if p.exists():
                for r in iter_jsonl(p):
                    score[a][r["question_id"]] = r["score_bool"]
                    del r
    for name, sel in [("trap (false-premise)", True), ("normal", False)]:
        ids = [q for q in lab if is_abs[q] == sel]
        c = Counter(lab[q] for q in ids)
        print(f"\n{name}: n={len(ids)}  " + "  ".join(f"{k}={c[k]} ({c[k]/len(ids):.2f})" for k in ["full", "partial", "none", "no-anchor"]))
        for k in ["full", "partial", "none", "no-anchor"]:
            S = [q for q in ids if lab[q] == k]
            if not S:
                continue
            cells = []
            for a in [ref] + arms:
                if all(q in score[a] for q in S):
                    cells.append(f"{a}={sum(score[a][q] for q in S)}/{len(S)}")
            print(f"   {k:9s} n={len(S):3d}  " + "  ".join(cells))
    print("\ntrap questions by type x coverage:")
    for t in sorted({qtype[q] for q in lab if is_abs[q]}):
        c = Counter(lab[q] for q in lab if is_abs[q] and qtype[q] == t)
        print(f"   {t:26s} " + "  ".join(f"{k}={c[k]}" for k in ["full", "partial", "none", "no-anchor"]))


if __name__ == "__main__":
    main()
