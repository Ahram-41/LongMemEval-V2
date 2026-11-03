#!/usr/bin/env python3
"""Readout of CP5101 experiments/b1-rt-resample-plan-2026-11-03.md (official baseline resampled on the 58
truncated questions). Scoring rule = score_uniform.load (capped -> forced wrap-up answer, else \\boxed{} required).

d = sum_q (rate_C(q) - rate_B1(q)): C over c1c_s1/s2, B1 over base + b1rt_s1..s3; 95% question bootstrap
(10,000, seed 0). Same d with C replaced by the regenerated-query official cut (c1a0_s1/s2), report only.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.oneoff.score_uniform import E1, ids, load  # noqa: E402

B1 = ["base", "b1rt_s1", "b1rt_s2", "b1rt_s3"]
C = ["c1c_s1", "c1c_s2"]
A0 = ["c1a0_s1", "c1a0_s2"]


def rates(vs, qs):
    arms = [load(v) for v in vs]
    for v, a in zip(vs, arms):
        miss = [q for q in qs if q not in a]
        assert not miss, f"{v}: {len(miss)} questions missing"
    return {q: sum(a[q]["ok"] for a in arms) / len(arms) for q in qs}, arms


def boot(diff, n=10_000, seed=0):
    rng = random.Random(seed)
    v = list(diff.values())
    s = sorted(sum(rng.choice(v) for _ in v) for _ in range(n))
    return s[int(0.025 * n)], s[int(0.975 * n) - 1]


def main() -> None:
    qs = sorted(set(load("c1c_s1")) & ids(E1 / "v2work/sys_Rt_ids.txt"))
    assert len(qs) == 58, len(qs)
    rb, ab = rates(B1, qs)
    rc, ac = rates(C, qs)
    ra, aa = rates(A0, qs)
    rep = {"n": len(qs)}
    for name, arms in (("B1", ab), ("C", ac), ("A0", aa)):
        rep[f"{name}_correct_per_sample"] = [sum(a[q]["ok"] for q in qs) for a in arms]
        rep[f"{name}_caps_per_sample"] = [sum(a[q]["cap"] for q in qs) for a in arms]
    for name, r in (("C_minus_B1", rc), ("A0_minus_B1", ra)):
        diff = {q: r[q] - rb[q] for q in qs}
        lo, hi = boot(diff)
        d = sum(diff.values())
        rep[name] = {"d": round(d, 2), "ci95": [round(lo, 2), round(hi, 2)]}
    d, (lo, hi) = rep["C_minus_B1"]["d"], rep["C_minus_B1"]["ci95"]
    rep["verdict"] = ("C above B1" if lo > 0 else "C below B1" if hi < 0 else
                      "parity" if abs(d) <= 3 else "inconclusive")
    rep["B1_0of4_C_2of2"] = [q for q in qs if rb[q] == 0 and rc[q] == 1]
    rep["B1_4of4_C_0of2"] = [q for q in qs if rb[q] == 1 and rc[q] == 0]
    out = E1 / "v2work/reports/b1rt_score.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
