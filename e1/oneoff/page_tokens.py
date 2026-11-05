#!/usr/bin/env python3
"""Where do the tokens of truncated questions go? (CP5101 experiments/trunc-shape-plan-2026-11-05.md §11; CPU only, report only)

For the questions of --ids-file, on the pre-truncation contexts (--contexts <dom>_contexts.jsonl), every raw-state AXTree line is
assigned to one part:
  boiler  : inside a landmark subtree (navigation, banner, contentinfo, search, complementary, menubar)
  rows    : inside a table/grid data row (a `row` whose direct children are not columnheaders)
  other   : everything else in the page (toolbar, forms, headers, detail fields, column headers)
and, across all parts, the formatting overhead that carries no page content:
  ids "[123] ", state flags (clickable, visible, focused, ...), and StaticText lines repeating their parent's name.
Tokens use the Qwen3.5-9B tokenizer. Per question: total prompt tokens (official count), raw-page tokens by part and by site,
tokens over the 197k budget, and whether dropping only data rows / only overhead would fit. Gold-answer phrases are located
by part (crude: short answers like 'D' match anywhere).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trunc_c  # noqa: E402
import boiler  # noqa: E402
from e1.common.io import iter_jsonl  # noqa: E402

LINE = re.compile(r"^(\t*)(\[[A-Za-z0-9]+\] )?([A-Za-z]+)(?: '((?:[^'\\]|\\.)*)')?(.*)$")
FLAGS = re.compile(r", (?:clickable|visible|focused|focusable|editable|hidden=False|disabled=False|required=False)\b")
BUDGET = trunc_c.MAX_TOKENS - trunc_c.MARGIN


def split_page(ax: str):
    """-> {part: [lines]}, compact text (overhead removed)."""
    lines = ax.split("\n")
    parsed = [LINE.match(l) for l in lines]
    depth = [len(m.group(1)) if m else 0 for m in parsed]
    role = [m.group(3) if m else "" for m in parsed]
    part = ["other"] * len(lines)
    i = 0
    while i < len(lines):
        if role[i] in trunc_c.LANDMARKS or role[i] == "row":
            j = i + 1
            while j < len(lines) and depth[j] > depth[i]:
                j += 1
            kids = {role[k] for k in range(i + 1, j) if depth[k] == depth[i] + 1}
            p = "boiler" if role[i] in trunc_c.LANDMARKS else (None if "columnheader" in kids else "rows")
            if p:
                part[i:j] = [p] * (j - i)
                i = j
                continue
        i += 1
    out, compact, parent_name = collections.defaultdict(list), [], {}
    for k, (l, m) in enumerate(zip(lines, parsed)):
        out[part[k]].append(l)
        if not m:
            compact.append(l)
            continue
        d, name = depth[k], m.group(4)
        parent_name[d] = name or ""
        if role[k] == "StaticText" and name is not None and d > 0 and name.strip() and name in parent_name.get(d - 1, ""):
            continue  # repeats the parent's accessible name
        compact.append(m.group(1) + m.group(3) + (f" '{name}'" if name is not None else "") + FLAGS.sub("", m.group(5)))
    return out, "\n".join(compact)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower().replace("-", " "))).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = trunc_c.tokenizer()
    n = lambda s: len(tok.encode(s).ids) if s else 0  # noqa: E731
    ids = set(Path(a.ids_file).expanduser().read_text().strip().split(","))
    rows = {}
    for dom in ("web", "enterprise"):
        for d in a.runs.split(","):
            p = trunc_c.E1 / f"runs/{d}/{dom}_small/prompt_rows.jsonl"
            if p.exists():
                for r in iter_jsonl(p):
                    if r["question_id"] in ids:
                        rows[r["question_id"]] = (dom, int(r["memory_context_original_token_count"]), r["answer_gold"])
    per_q = {}
    for dom in ("web", "enterprise"):
        for r in iter_jsonl(Path(a.contexts).expanduser() / f"{dom}_contexts.jsonl"):
            q = r["question_id"]
            if q not in rows or rows[q][0] != dom:
                continue
            _, total, gold = rows[q]
            parts, site_tok, nonraw, overhead, gold_in = collections.Counter(), collections.Counter(), 0, 0, collections.Counter()
            golds = [norm(g) for g in re.split(r"[,;]", gold) if norm(g)]
            for x in r["memory_context"]:
                if not trunc_c.is_raw(x):
                    nonraw += n(x.get("value", "")) if isinstance(x, dict) else 0
                    continue
                head, sts, _ = trunc_c.parse_block(x["value"])
                nonraw += n(head)
                for h, ax in sts:
                    out, compact = split_page(ax)
                    m = boiler.URL.search(h)
                    s = boiler.site(m.group(1) if m else "")
                    full = n(ax)
                    overhead += full - n(compact)
                    for p, ls in out.items():
                        t = n("\n".join(ls))
                        parts[p] += t
                        site_tok[f"{s}:{p}"] += t
                        txt = norm("\n".join(ls))
                        gold_in[p] += sum(g in txt for g in golds)
                    nonraw += n(h)
            raw = sum(parts.values())
            per_q[q] = {"dom": dom, "total": total, "over": total - BUDGET, "raw": raw, "nonraw_est": nonraw, "parts": dict(parts),
                        "overhead": overhead, "site_parts": dict(site_tok), "gold": gold, "gold_hits": dict(gold_in),
                        "fit_without_rows": total - parts["rows"] <= BUDGET,
                        "fit_without_overhead": total - overhead <= BUDGET,
                        "fit_without_magento_rows": total - site_tok.get("magento-admin:rows", 0) <= BUDGET}
    qs = list(per_q.values())
    S = lambda k: sum(x[k] for x in qs)  # noqa: E731
    raw = S("raw")
    summ = {"questions": len(qs), "total_tokens": S("total"), "raw_page_tokens": raw,
            "raw_parts_share": {p: round(sum(x["parts"].get(p, 0) for x in qs) / raw, 3) for p in ("boiler", "rows", "other")},
            "overhead_share_of_raw": round(S("overhead") / raw, 3),
            "over_budget_median": sorted(x["over"] for x in qs)[len(qs) // 2],
            "fit_without_rows": sum(x["fit_without_rows"] for x in qs),
            "fit_without_magento_rows": sum(x["fit_without_magento_rows"] for x in qs),
            "fit_without_overhead": sum(x["fit_without_overhead"] for x in qs),
            "by_dom": dict(collections.Counter(x["dom"] for x in qs))}
    st = collections.Counter()
    for x in qs:
        st.update(x["site_parts"])
    summ["site_part_share_of_raw"] = {k: round(v / raw, 3) for k, v in st.most_common(12)}
    dominant = collections.Counter()
    for x in qs:
        s = collections.Counter()
        for k, v in x["site_parts"].items():
            s[k.split(":")[0]] += v
        dominant[s.most_common(1)[0][0]] += 1
    summ["dominant_site_count"] = dict(dominant)
    loc = collections.Counter()
    for x in qs:
        h = x["gold_hits"]
        loc["none" if not any(h.values()) else "+".join(sorted(p for p, v in h.items() if v))] += 1
    summ["gold_phrase_found_in"] = dict(loc)
    Path(a.out).expanduser().write_text(json.dumps({"summary": summ, "per_q": per_q}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
