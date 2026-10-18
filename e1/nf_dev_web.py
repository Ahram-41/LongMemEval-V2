import json, re
split = "web_small"   # DEV ONLY
base = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"/home/w/wang41/e1/runs/agentrunbook_r_small/{split}/per_question.jsonl")}
ver = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"/home/w/wang41/e1/runs/p_small/{split}/premise_check.jsonl")}
QUOTED = re.compile(r'"([^"]{2,80})"|`([^`]{2,80})`|“([^”]{2,80})”|\'([^\']{3,80})\'')
def qterms(q): return {next(g for g in m.groups() if g).strip().lower() for m in QUOTED.finditer(q)}
def flagged(v, q, statuses, need_quote):
    if v["overall"] not in statuses: return False
    if not need_quote: return True
    t = qterms(q)
    return any(str(a.get("status","")).upper() in statuses and any(x in str(a.get("text","")).lower() for x in t) for a in v.get("assumptions", []))
nfix = sum(1 for q,r in base.items() if r["is_abstention_problem"] and not r["score_bool"])
nbrk = sum(1 for q,r in base.items() if not r["is_abstention_problem"] and r["score_bool"])
for name, st, nq in [("CONTRA", {"CONTRADICTED"}, False), ("CONTRA+quote", {"CONTRADICTED"}, True),
                     ("CONTRA|NF", {"CONTRADICTED","NOT_FOUND"}, False), ("CONTRA|NF+quote", {"CONTRADICTED","NOT_FOUND"}, True)]:
    fix = sum(1 for q,r in base.items() if r["is_abstention_problem"] and not r["score_bool"] and flagged(ver[q], r["question_text"], st, nq))
    brk = sum(1 for q,r in base.items() if not r["is_abstention_problem"] and r["score_bool"] and flagged(ver[q], r["question_text"], st, nq))
    print(f"{split} {name:16s} could-fix {fix:2d}/{nfix} ({fix/nfix:.2f})  could-break {brk:2d}/{nbrk} ({brk/nbrk:.2f})  fix/break={fix/max(brk,1):.2f}")
