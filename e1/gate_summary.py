#!/usr/bin/env python3
"""Gate for arm P: does the premise checker separate false-premise from normal questions?

Joins <arm>/<dom>_small/premise_check.jsonl with the baseline per_question.jsonl
(labels used only here, for evaluation; the checker never saw them).
Usage: python e1/gate_summary.py ~/e1/runs/agentrunbook_r_small ~/e1/runs/p_small
"""
import json, statistics, sys
from collections import Counter
from pathlib import Path

base, arm = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
rows = {}
for d in sorted(base.glob("*_small")):
    for l in open(d / "per_question.jsonl"):
        r = json.loads(l); rows[r["question_id"]] = r
ver = {}
for d in sorted(arm.glob("*_small")):
    p = d / "premise_check.jsonl"
    if p.exists():
        for l in open(p):
            v = json.loads(l); ver[v["question_id"]] = v
print(f"verdicts: {len(ver)} / {len(rows)} questions")
for name, sel in [("false-premise (abstention)", lambda r: r["is_abstention_problem"]),
                  ("normal (non-abstention)", lambda r: not r["is_abstention_problem"])]:
    ids = [q for q, r in rows.items() if sel(r) and q in ver]
    c = Counter(ver[q]["overall"] for q in ids)
    print(f"\n{name}: n={len(ids)}")
    for k in ["CONTRADICTED", "NOT_FOUND", "SUPPORTED", "PARSE_ERROR", "CALL_ERROR"]:
        if c[k]: print(f"  {k:13s} {c[k]:4d}  ({c[k]/len(ids):.3f})")
    if name.startswith("normal"):
        ok = [q for q in ids if rows[q]["score_bool"]]
        flagged_ok = sum(ver[q]["overall"] == "CONTRADICTED" for q in ok)
        print(f"  baseline-correct normal questions flagged CONTRADICTED: {flagged_ok}/{len(ok)} ({flagged_ok/max(len(ok),1):.3f})  <- answers P could break")
    else:
        wrong = [q for q in ids if not rows[q]["score_bool"]]
        flagged_wrong = sum(ver[q]["overall"] == "CONTRADICTED" for q in wrong)
        print(f"  baseline-wrong false-premise questions flagged CONTRADICTED: {flagged_wrong}/{len(wrong)} ({flagged_wrong/max(len(wrong),1):.3f})  <- answers P could fix")
secs = [v["check_seconds"] for v in ver.values() if v.get("check_seconds")]
if secs:
    print(f"\nchecker seconds/question: mean {statistics.mean(secs):.1f}, median {statistics.median(secs):.1f}, max {max(secs):.1f}")
lat = []
for d in sorted(arm.glob("*_small")):
    p = d / "premise_check_latency.jsonl"
    if p.exists():
        lat += [json.loads(l)["check_seconds"] for l in open(p)]
if lat:
    print(f"sequential (one-at-a-time) checker seconds: n={len(lat)} mean {statistics.mean(lat):.1f} median {statistics.median(lat):.1f} max {max(lat):.1f}")
    print(f"conservative P query latency = baseline 13.9 s (measured under 8-way concurrency) + {statistics.mean(lat):.1f} = {13.9+statistics.mean(lat):.1f} s vs budget 26.9 s")
