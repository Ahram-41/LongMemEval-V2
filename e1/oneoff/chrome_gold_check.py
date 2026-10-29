"""For trad breaks/fixes: is a gold phrase only in chrome lines that trad moved to the top list?"""
import json, re, sys
sys.path.insert(0, "e1/oneoff")
from reader_interventions import load_rows, DOMS
from trad_steps import build_chrome, trad_messages, text_of, norm
ids = sys.argv[1].split(",")
for dom in DOMS:
    rows = load_rows(dom)
    chrome = build_chrome([r["messages"] for r in rows.values()])
    for q, r in rows.items():
        if not any(q.startswith(i) for i in ids):
            continue
        new, _ = trad_messages(r["messages"], chrome)
        t0, t1 = text_of(r["messages"]).lower(), text_of(new).lower()
        head = t1.split("### common page elements", 1)
        clist = head[1].split("### raw state result", 1)[0] if len(head) > 1 else ""
        body = t1.replace(clist, "")
        out = []
        for ph in [x.strip().lower() for x in re.split(r"[,;]", r["answer_gold"]) if x.strip()]:
            out.append(f"{ph[:30]!r}: orig {t0.count(ph)} trad {t1.count(ph)} in-chrome-list {clist.count(ph)} body {body.count(ph)}")
        print(q[:8], dom[:3], "| " + " ; ".join(out))
