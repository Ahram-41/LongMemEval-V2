#!/usr/bin/env python3
"""Per raw block of the RR pre-truncation context: URL per state, present in RR (official truncation)?,
kept chars in RR+C, and question-specific keyword hits (offset vs the C cut). Companion of c2_fix_positions.py."""
from __future__ import annotations
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_keys  # noqa: E402
from trunc_c import is_raw, parse_block  # noqa: E402
from e1.common.io import iter_jsonl  # noqa: E402
E1 = step_keys.E1
KW = {"15e5efa6": ["export"], "597ca614": ["per page"], "7c8efee5": ["per page"],
      "44c05e59": ["upvoted", "downvoted", "saved", "hidden", "submissions"],
      "85fc5722": ["access type"], "9e05978c": ["short description", "show filter", "series-df9c3c1e"]}
CUT = re.compile(r"\n\.\.\.\[truncated (\d+) chars\]\.\.\.\n$")

def blocks(mc):
    out = {}
    for i, x in enumerate(mc):
        if is_raw(x):
            try: out[i] = parse_block(x["value"])[1]
            except Exception: out[i] = None
    return out

for dom in step_keys.DOMS:
    from e1.pipeline.retrieve import meta
    M = meta(dom)
    qs = {q: w for q in M for w in KW if q.startswith(w)}
    if not qs: continue
    pre = {r["question_id"]: r["memory_context"] for r in iter_jsonl(E1 / f"e2work/contexts/rr/{dom}_contexts.jsonl") if r["question_id"] in qs}
    rr = {r["question_id"]: r["memory_context"] for r in iter_jsonl(E1 / f"runs/rr_small/{dom}_small/prompt_rows.jsonl") if r["question_id"] in qs}
    rc = {r["question_id"]: r["memory_context"] for r in iter_jsonl(E1 / f"runs/c2_rc_small/{dom}_small/prompt_rows.jsonl") if r["question_id"] in qs}
    for q in sorted(qs):
        w = qs[q]; print(f"\n=== {w} [{dom}]\nQ: {M[q]['question_text'][:900]}\ngold: {M[q]['answer_gold']}")
        P, R, C = blocks(pre[q]), blocks(rr[q]), blocks(rc[q])
        for i, sts in P.items():
            inrr = "RR:yes" if i in R and R[i] is not None and len(R[i]) == len(sts) and all(a[1] == b[1] for a, b in zip(R[i], sts)) else ("RR:partial" if i in R else "RR:dropped")
            for k, (h, ax) in enumerate(sts):
                url = re.search(r"- URL: (\S+)", h); cax = C[i][k][1] if C.get(i) else ""
                m = CUT.search(cax); kept = m.start() if m else len(cax)
                low = ax.lower(); hs = []
                for kw in KW[w]:
                    offs = [mm.start() for mm in re.finditer(re.escape(kw), low)]
                    if offs:
                        hs.append(f"{kw}@{offs[0]}" + (f"(+{len(offs)-1})" if len(offs) > 1 else "") + ("✔" if offs[0] < kept else "✘cut"))
                print(f" it{i:<2} s{k} {inrr:10} len{len(ax):>7} keptC{kept:>7} {url.group(1)[:70] if url else '?':70} {' '.join(hs)}")
        # snippets of the first hit per keyword
        for kw in KW[w]:
            for i, sts in P.items():
                done = False
                for k, (h, ax) in enumerate(sts):
                    j = ax.lower().find(kw)
                    if j >= 0:
                        print(f"  snip it{i} s{k} {kw!r}: " + ax[max(0, j-150): j+250].replace("\n", " | ")); done = True; break
                if done: break
