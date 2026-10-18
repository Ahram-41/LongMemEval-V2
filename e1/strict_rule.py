"""P-strict rule, fixed on the WEB split only (dev). Do not run on enterprise until frozen."""
import json, re, sys
split = sys.argv[1]  # web_small (dev) or enterprise_small (held-out)
base = {}
for l in open(f"/home/w/wang41/e1/runs/agentrunbook_r_small/{split}/per_question.jsonl"):
    r = json.loads(l); base[r["question_id"]] = r
ver = [json.loads(l) for l in open(f"/home/w/wang41/e1/runs/p_small/{split}/premise_check.jsonl")]
QUOTED = re.compile(r'"([^"]{2,80})"|`([^`]{2,80})`|“([^”]{2,80})”|\'([^\']{3,80})\'')
def quoted_terms(q):
    return {next(g for g in m.groups() if g).strip().lower() for m in QUOTED.finditer(q)}
def strict(v, q):
    if v["overall"] != "CONTRADICTED": return False
    terms = quoted_terms(q)
    for a in v.get("assumptions", []):
        if str(a.get("status", "")).upper() != "CONTRADICTED": continue
        txt = str(a.get("text", "")).lower()
        if any(t in txt for t in terms): return True
    return False
rows = []
for v in ver:
    r = base[v["question_id"]]
    rows.append((r["is_abstention_problem"], r["score_bool"], v["overall"] == "CONTRADICTED", strict(v, r["question_text"])))
def report(idx, name):
    fix = sum(1 for a, s, *f in rows if a and not s and f[idx]); nfix = sum(1 for a, s, *_ in rows if a and not s)
    brk = sum(1 for a, s, *f in rows if not a and s and f[idx]); nbrk = sum(1 for a, s, *_ in rows if not a and s)
    print(f"{split} {name:10s} could-fix {fix}/{nfix} ({fix/nfix:.3f})   could-break {brk}/{nbrk} ({brk/nbrk:.3f})")
report(0, "P (any)"); report(1, "P-strict")
