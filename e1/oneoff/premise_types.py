import json, re
from collections import Counter, defaultdict
rows = [json.loads(l) for l in open("questions.jsonl")]
abs_r = [r for r in rows if r["question_type"].endswith("-abs")]
def text(r):
    q = r["question"]; return q if isinstance(q, str) else " ".join(x.get("value", "") for x in q if isinstance(x, dict) and x.get("type") == "text")
RULES = [  # (type, regex on gold answer), first match wins
    ("count overreach (asks for an Nth / extra item that does not exist)", r"\bonly (one|two|three|\d)\b|no (second|third|fourth|other|additional|extra)\b|besides\b"),
    ("adjacency / position (nothing lies between or at that position)", r"\bbetween\b|adjacent|next to each other|directly (below|above|after)"),
    ("unsupported feature / impossible action", r"not (support|achievable|possible)|does not support|impossible|cannot be|can't be|no way to"),
    ("missing element (the named field, option, link, tab does not exist)", r"\bthere (is|are) no\b|\bno such\b|does not (exist|have|appear|expose|show|contain)|not (present|shown|listed|available)|lacks?\b|neither\b"),
    ("wrong value / state (the element exists but is different)", r"not (pre|already)|empty|blank|no (value|default|prefix|text)|unchanged|does not change|stays|remains"),
]
cnt = Counter(); by_qt = defaultdict(Counter); ex = defaultdict(list)
for r in abs_r:
    g = r["answer"].lower(); t = "other / unclear"
    for name, rx in RULES:
        if re.search(rx, g): t = name; break
    cnt[t] += 1; by_qt[r["question_type"]][t] += 1
    if len(ex[t]) < 2: ex[t].append((text(r)[:150], r["answer"][:120]))
print(f"false-premise questions: {len(abs_r)}")
for t, n in cnt.most_common():
    print(f"  {n:3d} ({n/len(abs_r):.2f})  {t}")
print("\nby question type:")
for qt in sorted(by_qt): print(f"  {qt:24s}", dict(by_qt[qt].most_common()))
print("\nexamples:")
for t, xs in ex.items():
    for q, a in xs[:1]: print(f"- [{t[:30]}] Q: {q}\n     gold: {a}")
