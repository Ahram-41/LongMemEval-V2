import json, sys
arm, dom = sys.argv[1], sys.argv[2]
b0 = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"/home/w/wang41/e1/runs/agentrunbook_r_small/{dom}/per_question.jsonl")}
p = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"/home/w/wang41/e1/runs/{arm}/{dom}/per_question.jsonl")}
cap = lambda r: (r.get("usage") or {}).get("completion_tokens", 0) >= 20000
for name, sel in [("false-premise", True), ("normal", False)]:
    ids = [q for q in p if b0[q]["is_abstention_problem"] == sel]
    if not ids: continue
    a0 = sum(b0[q]["score_bool"] for q in ids); a1 = sum(p[q]["score_bool"] for q in ids)
    fixed = sum(not b0[q]["score_bool"] and p[q]["score_bool"] for q in ids)
    broke = sum(b0[q]["score_bool"] and not p[q]["score_bool"] for q in ids)
    c0 = sum(cap(b0[q]) for q in ids); c1 = sum(cap(p[q]) for q in ids)
    print(f"{arm} {dom} {name:13s} noted n={len(ids):3d}  B0 correct {a0:3d} -> {arm} {a1:3d}  (fixed {fixed}, broke {broke})  capped {c0} -> {c1}")
