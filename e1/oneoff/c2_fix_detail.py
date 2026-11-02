#!/usr/bin/env python3
"""Follow-ups to c2_fix_evidence.py: 9e05978c dropped-block novelty, 85fc5722 sort-row snippet, 44c05e59 profile pages."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_keys  # noqa: E402
from trunc_c import is_raw, parse_block  # noqa: E402
from e1.common.io import iter_jsonl  # noqa: E402
E1 = step_keys.E1
def ctx(dom, q8):
    return next((r["question_id"], r["memory_context"]) for r in iter_jsonl(E1 / f"e2work/contexts/rr/{dom}_contexts.jsonl") if r["question_id"].startswith(q8))
# 9e05978c: are the it39 states (dropped in RR) and the cut tails novel w.r.t. what RR keeps?
q, mc = ctx("enterprise", "9e05978c")
B = {i: parse_block(x["value"]) for i, x in enumerate(mc) if is_raw(x)}
kept_lines = set()
for i, (_, sts, _) in B.items():
    if i < 39:
        for h, ax in sts: kept_lines.update(l.strip() for l in ax.splitlines())
for k, (h, ax) in enumerate(B[39][1]):
    same = [f"it{i}s{j}" for i, (_, s2, _) in B.items() if i < 39 for j, (_, a2) in enumerate(s2) if a2 == ax]
    new = [l.strip() for l in ax.splitlines() if l.strip() and l.strip() not in kept_lines]
    print(f"9e05978c it39 s{k}: identical to {same or 'none'}; {len(new)} novel lines; e.g. {new[:6]}")
print("9e05978c it39 head:", B[39][0][:600].replace("\n", " | "))
rc = {r["question_id"]: r for r in iter_jsonl(E1 / "runs/c2_rc_small/enterprise_small/prompt_rows.jsonl") if r["question_id"] == q}
rr = {r["question_id"]: r for r in iter_jsonl(E1 / "runs/rr_small/enterprise_small/prompt_rows.jsonl") if r["question_id"] == q}
for nm, r in [("rr", rr[q]), ("rc", rc[q])]:
    print(nm, "orig_tokens", r.get("memory_context_original_token_count"), "truncated", r.get("memory_context_was_truncated"),
          "chars", sum(len(x.get("value", "")) for x in r["memory_context"]))
# 85fc5722 it35 s0 around 'access type', and what the RR-kept blocks say about sort
q, mc = ctx("enterprise", "85fc5722")
B = {i: parse_block(x["value"]) for i, x in enumerate(mc) if is_raw(x)}
ax = B[35][1][0][1]; j = ax.lower().find("access type")
print("\n85fc5722 it35 head:", B[35][0][:500].replace("\n", " | "))
print("85fc5722 it35 s0:", ax[max(0, j-500): j+200].replace("\n", " | "))
for i in (21, 29):
    print(f"85fc5722 it{i} head:", B[i][0][:400].replace("\n", " | "))
    for k, (h, a) in enumerate(B[i][1]):
        for kw in ("sort", "order by", "access type"):
            jj = a.lower().find(kw)
            if jj >= 0: print(f"   s{k} {kw!r}@{jj}: " + a[max(0, jj-120): jj+160].replace("\n", " | "))
# 44c05e59: which reddit pages exist and the block heads
q, mc = ctx("web", "44c05e59")
for i, x in enumerate(mc):
    if is_raw(x):
        hd, sts, _ = parse_block(x["value"])
        urls = [re.search(r"- URL: (\S+)", h).group(1) for h, _ in sts]
        if any(":9080" in u for u in urls):
            print(f"\n44c05e59 it{i} head:", hd[:500].replace("\n", " | ")); print("  urls:", urls)
            for k, (h, a) in enumerate(sts):
                if "/user/" in urls[k]:
                    j = a.find("link 'Overview'"); print("  tabs:", a[max(0, j-200): j+400].replace("\n", " | "))
# B1 stability + C2 per-q for all six
rep = json.loads((step_keys.V2 / "reports/trunc_reader_c2only.json").read_text())["C2"]["per_q"]
for q8 in ["15e5efa6", "44c05e59", "597ca614", "7c8efee5", "85fc5722", "9e05978c"]:
    print(q8, next(v for k, v in rep.items() if k.startswith(q8)))
