#!/usr/bin/env python3
"""Question partition for the system-level C arm (CP5101 experiments/sys-c-plan-2026-11-03.md).

  U   : rule-scored questions the B1 context did not truncate (B1 prompt rows are complete; K and T both use them)
  R_u : B1-truncated questions whose regenerated-query (A0) context fits in 200k (K and T both use A0 rows)
  R_t : B1-truncated questions A0 also truncates (K = A0 official cut = c1_a0, T = C = c1_c)

Writes ~/e1/v2work/sys_{U,Ru,Rt}_ids.txt (comma separated, all questions; the reader keeps the rule-scored
ones) and prints the counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.oneoff.reader_interventions import E1, rule_scored  # noqa: E402

DOMS = ("web", "enterprise")
V2 = E1 / "v2work"


def rows(run: str, dom: str):
    p = E1 / f"runs/{run}/{dom}_small/prompt_rows.jsonl"
    with open(p) as fp:
        for line in fp:
            if line.strip():
                r = json.loads(line)
                yield r["question_id"], str(r["memory_context_was_truncated"]) == "True", rule_scored(r)
                del r


def main() -> None:
    U, R, Ru, Rt, scored = set(), set(), set(), set(), set()
    for dom in DOMS:
        for q, trunc, sc in rows("b1_small", dom):
            (R if trunc else U).add(q)
            if sc:
                scored.add(q)
        for run in ("e2_ordA0_small", "e2_ordA0rest_small"):
            for q, trunc, _ in rows(run, dom):
                (Rt if trunc else Ru).add(q)
    c1 = set((V2 / "c1_ids.txt").read_text().replace(",", " ").split())
    assert Rt == c1, (len(Rt), len(c1))
    assert Ru | Rt == R and not Ru & Rt, (len(R), len(Ru), len(Rt))
    for name, s in (("U", U), ("Ru", Ru), ("Rt", Rt)):
        (V2 / f"sys_{name}_ids.txt").write_text(",".join(sorted(s)))
        print(f"{name}: {len(s)} questions, {len(s & scored)} rule-scored")


if __name__ == "__main__":
    main()
