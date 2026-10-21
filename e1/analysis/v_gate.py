#!/usr/bin/env python3
"""E2 gate for V: recompute the official dense search with the frozen (parsed) queries and
recomputed query vectors, apply the official merge, and compare with the B0 raw blocks.
Only questions whose queries were fully recovered (context not truncated, no merge-implied
missing query) are compared. Reports exact equality of the (entry, query, rank) sequence, of the
entry set, and the largest |similarity| difference.
Usage: E2WORK=~/e1/e2work_vgate python -I e1/analysis/v_gate.py --domains web,enterprise
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.pipeline.retrieve import E1, do_score, do_select  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="web,enterprise")
    a = ap.parse_args()
    for dom in a.domains.split(","):
        do_score(dom, ["V"])
        rows = {r["question_id"]: r for r in do_select(dom, "V", "frozen", None)}
        base = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl")}
        n = seq = sets = 0
        dmax, bad = 0.0, []
        for qid, b in base.items():
            if b["context_truncated"] or b["missing_detected"]:
                continue
            n += 1
            mine = [(x["entry_id"], x["query"], x["rank"]) for x in rows[qid]["raw_blocks"]]
            theirs = [(x["entry_id"], x["query"], x["rank"]) for x in b["baseline_raw_blocks"]]
            seq += mine == theirs
            sets += {m[0] for m in mine} == {t[0] for t in theirs}
            if mine == theirs:
                dmax = max([dmax] + [abs(x["score"] - y["score"]) for x, y in zip(rows[qid]["raw_blocks"], b["baseline_raw_blocks"])])
            else:
                bad.append(qid)
        print(f"[V gate] {dom}: identical block sequence {seq}/{n} ({seq/n:.3f}); identical block set {sets}/{n}; "
              f"max |similarity diff| on identical ones {dmax:.5f}; differing: {bad[:8]}")


if __name__ == "__main__":
    main()
