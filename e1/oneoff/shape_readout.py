#!/usr/bin/env python3
"""Report-only breakdown of the cut-shape test (CP5101 experiments/trunc-shape-plan-2026-11-05.md §4):
per treatment vs per-page head cut, correct counts (of 2 samples) grouped by the question's dominant site
(most raw-page characters, from boiler_rr.json), question type, the 4 toolbar questions and the 12 questions
that fit after boilerplate removal. Input: ~/e1/v2work/reports/trunc_shape.json (trunc_c.py score --pairs).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

V2 = Path("~/e1/v2work").expanduser()
TOOLBAR = ["15e5efa6", "597ca614", "7c8efee5", "85fc5722"]


def main() -> None:
    shape = json.loads((V2 / "reports/trunc_shape.json").read_text())
    boil = json.loads((V2 / "reports/boiler_rr.json").read_text())
    fit = set(boil["summary"]["truncated_fit_after_boiler_removal"])
    info = {}
    for q, i in boil["per_q"].items():
        chars = collections.Counter()
        for s in i["states"]:
            chars[s["site"]] += s["chars"]
        info[q] = {"site": chars.most_common(1)[0][0] if chars else "?", "qtype": i["qtype"]}
    for name, r in shape.items():
        pq = r["per_q"]
        print(f"\n== {name}: control {r['control_correct']} treatment {r['treatment_correct']} of {2 * r['n']}, "
              f"stable fixed {r['stable_fixed']} broken {r['stable_broken']} net {r['net']:+d} -> {r['verdict']}; caps {r['caps']}")
        for key in ("site", "qtype"):
            g = collections.defaultdict(lambda: [0, 0, 0])
            for q, (a, b, _) in pq.items():
                k = info.get(q, {}).get(key, "?")
                g[k][0] += 1
                g[k][1] += a
                g[k][2] += b
            print(f"  by {key}: " + "; ".join(f"{k} n={v[0]} {v[1]}->{v[2]}" for k, v in sorted(g.items(), key=lambda kv: -kv[1][0])))
        print("  toolbar 4: " + ", ".join(f"{q} {pq[q][0]}->{pq[q][1]}" for q in TOOLBAR if q in pq))
        inside = [q for q in pq if q in fit]
        print(f"  fit-after-boiler {len(inside)}: {sum(pq[q][0] for q in inside)}->{sum(pq[q][1] for q in inside)}; "
              f"others {len(pq) - len(inside)}: {sum(v[0] for q, v in pq.items() if q not in fit)}->{sum(v[1] for q, v in pq.items() if q not in fit)}")
        moved = [f"{q}({info.get(q, {}).get('site', '?')}) {a}->{b}" for q, (a, b, _) in sorted(pq.items()) if abs(b - a) == 2]
        print("  0<->2 moves: " + ", ".join(moved))


if __name__ == "__main__":
    main()
