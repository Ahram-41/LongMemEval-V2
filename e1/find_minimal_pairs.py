import json, re
rows = [json.loads(l) for l in open("questions.jsonl")]
def text(r):
    q = r["question"]; return q if isinstance(q, str) else " ".join(x.get("value", "") for x in q if isinstance(x, dict) and x.get("type") == "text")
FMT = re.compile(r"(mark|wrap|make sure|your final answer|answer with|please answer|give me)[^.]*\.?", re.I)
def core(r): return set(re.findall(r"[a-z]+", FMT.sub(" ", text(r)).lower()))
abs_r = [r for r in rows if r["question_type"].endswith("-abs")]
norm_r = [r for r in rows if not r["question_type"].endswith("-abs")]
best = []
for a in abs_r:
    ca = core(a); cand = [(len(ca & core(n)) / max(len(ca | core(n)), 1), n) for n in norm_r if n["domain"] == a["domain"]]
    s, n = max(cand, key=lambda x: x[0]); best.append((s, a, n))
best.sort(key=lambda x: -x[0])
import statistics
sims = [b[0] for b in best]
print(f"false-premise questions: {len(best)}; best-match Jaccard to a normal question: median {statistics.median(sims):.2f}")
for th in [0.8, 0.6, 0.5, 0.4]:
    print(f"  pairs with Jaccard >= {th}: {sum(s >= th for s in sims)}")
same_type = sum(1 for s, a, n in best if s >= 0.5 and n["question_type"] == a["question_type"].replace("-abs", ""))
print(f"  of pairs >= 0.5, same base type: {same_type}")
for s, a, n in best[:3] + best[60:62]:
    print(f"\nJ={s:.2f}\n  FALSE: {text(a)[:200]}\n  NORMAL: {text(n)[:200]}")
json.dump([{"abs_id": a["id"], "normal_id": n["id"], "jaccard": s} for s, a, n in best], open("pairs.json", "w"), indent=1)
