#!/usr/bin/env python3
"""Compare arms on the P-any noted questions only (paired, exact McNemar), split by
false-premise / normal, plus cap-hit rate and, for mid-injection arms, how many were injected.
Usage: python e1/noted_compare.py ~/e1/runs b1_small pa_small mi_small mn_small
The first arm is the reference. Rows for B1 come from the full run; noted arms only hold noted rows."""
import json, math, sys
from pathlib import Path
root = Path(sys.argv[1]).expanduser(); arms = sys.argv[2:]
KEEP = ("question_id", "score_bool", "is_abstention_problem", "usage")
def load(arm):
    out = {}
    for d in ["web_small", "enterprise_small"]:
        p = root / arm / d / "per_question.jsonl"
        if p.exists():
            for l in open(p):
                r = json.loads(l); out[r["question_id"]] = {k: r.get(k) for k in KEEP}; del r
    return out
def inj(arm):
    s = {}
    for d in ["web_small", "enterprise_small"]:
        p = root / arm / d / "mid_inject_log.jsonl"
        if p.exists():
            for l in open(p):
                x = json.loads(l); s[x["question_id"]] = x
    return s
ids = []
for d in ["web_small", "enterprise_small"]:
    ids += [x for x in (root / "pa_small" / d / "noted_ids.txt").read_text().split(",") if x]
D = {a: load(a) for a in arms}; ref = arms[0]
ids = [q for q in ids if all(q in D[a] for a in arms)]
def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
cap = lambda r: (r.get("usage") or {}).get("completion_tokens", 0) >= 20000
print(f"noted questions with all arms present: {len(ids)}")
for name, sel in [("false-premise", True), ("normal", False), ("all noted", None)]:
    S = [q for q in ids if sel is None or D[ref][q]["is_abstention_problem"] == sel]
    print(f"\n{name} (n={len(S)})")
    for a in arms:
        acc = sum(D[a][q]["score_bool"] for q in S); cp = sum(cap(D[a][q]) for q in S)
        line = f"  {a:14s} correct {acc:3d} ({acc/len(S):.3f})  capped {cp:3d}"
        if a != ref:
            b = sum(D[ref][q]["score_bool"] and not D[a][q]["score_bool"] for q in S)
            c = sum((not D[ref][q]["score_bool"]) and D[a][q]["score_bool"] for q in S)
            line += f"  vs {ref}: fixed {c}, broke {b}, McNemar p={mcn(b, c):.3f}"
        print(line)
for a in arms:
    L = inj(a)
    if L:
        I = [q for q in ids if L.get(q, {}).get("injected")]
        fp = [q for q in I if D[ref][q]["is_abstention_problem"]]; nm = [q for q in I if not D[ref][q]["is_abstention_problem"]]
        print(f"\n{a}: injected on {len(I)}/{len(ids)} (false-premise {len(fp)}, normal {len(nm)})")
        for nm_, S in [("injected false-premise", fp), ("injected normal", nm)]:
            if S:
                print(f"  {nm_:24s} {ref} {sum(D[ref][q]['score_bool'] for q in S)}/{len(S)}  ->  {a} {sum(D[a][q]['score_bool'] for q in S)}/{len(S)}"
                      + (f"   (pa_small {sum(D['pa_small'][q]['score_bool'] for q in S)}/{len(S)})" if 'pa_small' in D else ""))
