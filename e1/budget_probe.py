#!/usr/bin/env python3
"""Budget probe: questions that hit the 20k thinking cap in B0, rerun with the B1 prompt and a 40k cap.

Compares, on the same questions, B0 (20k, buggy prompt), B1 (20k, fixed prompt) when available,
and bud (40k, fixed prompt). Question: does more thinking budget rescue capped false-premise
questions (slow convergence) or not (stuck)?
Usage: python e1/budget_probe.py ~/e1/runs
"""
import json, sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
def load(arm):
    out = {}
    for d in ["web_small", "enterprise_small"]:
        p = root / arm / d / "per_question.jsonl"
        if p.exists():
            for l in open(p):
                r = json.loads(l); out[r["question_id"]] = r
    return out
b0, b1, bud = load("agentrunbook_r_small"), load("b1_small"), load("bud_small")
toks = lambda r: (r.get("usage") or {}).get("completion_tokens", 0)
print(f"bud questions scored: {len(bud)}")
for name, sel in [("false-premise", True), ("normal", False)]:
    ids = [q for q in bud if b0[q]["is_abstention_problem"] == sel]
    if not ids: continue
    a0 = sum(b0[q]["score_bool"] for q in ids)
    ab = sum(bud[q]["score_bool"] for q in ids)
    still = sum(toks(bud[q]) >= 40000 for q in ids)
    med = sorted(toks(bud[q]) for q in ids)[len(ids) // 2]
    line = f"{name:13s} n={len(ids):3d}  B0@20k correct {a0:3d}  bud@40k correct {ab:3d}  still capped at 40k {still:3d}  median tokens {med}"
    common = [q for q in ids if q in b1]
    if common:
        line += f"  | B1@20k on {len(common)} of them: {sum(b1[q]['score_bool'] for q in common)} correct"
    print(line)
