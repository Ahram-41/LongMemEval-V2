#!/usr/bin/env python3
"""Boilerplate share of raw page states (CP5101 experiments/boiler-trunc-plan-2026-11-05.md, step 1; CPU only).

Boilerplate = AXTree subtrees rooted at a landmark role (navigation, banner, contentinfo, search, complementary,
menubar, LANDMARKS), i.e. site menus / headers / footers / search bars; everything else (main, tables, forms) is kept.
AXTree lines are tab-indented "[id] role 'name', ..." (or "role '...'" without id); a subtree ends at the next line
with depth <= its root.

  measure : for the questions of a run's prompt rows, on the pre-truncation contexts: per state chars, boilerplate
            chars by role, site (URL), and per question the official token count minus the boilerplate tokens
            (Qwen3.5-9B tokenizer.json) -> would the context fit the 200k budget without any truncation?
            -> ~/e1/v2work/reports/boiler_<name>.json + printed summary
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
from e1.common.io import iter_jsonl  # noqa: E402

E1 = trunc_c.E1
LANDMARKS, strip_boiler = trunc_c.LANDMARKS, trunc_c.strip_boiler
URL = re.compile(r"- URL: (\S+)")


def site(url: str) -> str:
    u = url.lower()
    for key, name in (("/admin", "magento-admin"), (":7780", "magento-admin"), (":9999", "reddit"), ("postmill", "reddit"),
                      (":8023", "gitlab"), ("gitlab", "gitlab"), (":7770", "shopping"), (":3000", "map"), ("wiki", "wiki")):
        if key in u:
            return name
    m = re.match(r"https?://([^/]+)", u)
    return m.group(1) if m else "?"


def measure(name: str, runs: list[str], contexts: Path, ids: set | None) -> None:
    tok = trunc_c.tokenizer()
    rows, per_q = {}, {}
    for dom in ("web", "enterprise"):
        for d in runs:
            p = E1 / f"runs/{d}/{dom}_small/prompt_rows.jsonl"
            if p.exists():
                for r in iter_jsonl(p):
                    if ids is None or r["question_id"] in ids:
                        rows[r["question_id"]] = {"dom": dom, "orig": int(r["memory_context_original_token_count"]),
                                                  "trunc": str(r["memory_context_was_truncated"]) == "True",
                                                  "qtype": r.get("question_type"), "eval": r.get("eval_name")}
        role_chars = collections.Counter()
        for r in iter_jsonl(contexts / f"{dom}_contexts.jsonl"):
            q = r["question_id"]
            if q not in rows or rows[q]["dom"] != dom:
                continue
            tot = boil = rm_tok = 0
            states = []
            for x in r["memory_context"]:
                if not trunc_c.is_raw(x):
                    continue
                _, sts, _ = trunc_c.parse_block(x["value"])
                for h, ax in sts:
                    kept, rem = strip_boiler(ax)
                    b = sum(rem.values())
                    role_chars.update(rem)
                    tot += len(ax)
                    boil += b
                    if b:
                        rm_tok += len(tok.encode(ax).ids) - len(tok.encode(kept).ids)
                    m = URL.search(h)
                    states.append({"site": site(m.group(1) if m else ""), "chars": len(ax), "boiler": b})
            info = rows[q]
            info.update({"raw_chars": tot, "boiler_chars": boil, "boiler_tokens": rm_tok,
                         "fits_after": info["orig"] - rm_tok <= trunc_c.MAX_TOKENS - trunc_c.MARGIN, "states": states})
            per_q[q] = info
        print(f"[boiler] {name} {dom}: removed chars by root role {dict(role_chars.most_common())}", flush=True)
    by_site = collections.defaultdict(lambda: [0, 0, 0])
    for info in per_q.values():
        for s in info["states"]:
            by_site[s["site"]][0] += 1
            by_site[s["site"]][1] += s["chars"]
            by_site[s["site"]][2] += s["boiler"]
    tr = [q for q, i in per_q.items() if i["trunc"]]
    summary = {
        "questions": len(per_q), "truncated": len(tr),
        "truncated_fit_after_boiler_removal": sorted(q for q in tr if per_q[q]["fits_after"]),
        "boiler_share_all": round(sum(i["boiler_chars"] for i in per_q.values()) / max(1, sum(i["raw_chars"] for i in per_q.values())), 3),
        "boiler_share_truncated": round(sum(per_q[q]["boiler_chars"] for q in tr) / max(1, sum(per_q[q]["raw_chars"] for q in tr)), 3),
        "by_site_states_chars_boilershare": {k: [v[0], v[1], round(v[2] / max(1, v[1]), 3)] for k, v in sorted(by_site.items(), key=lambda kv: -kv[1][1])},
        "truncated_excess_vs_boiler_tokens": {q: [per_q[q]["orig"] - (trunc_c.MAX_TOKENS - trunc_c.MARGIN), per_q[q]["boiler_tokens"]] for q in sorted(tr)},
    }
    out = trunc_c.step_keys.V2 / f"reports/boiler_{name}.json"
    out.write_text(json.dumps({"summary": summary, "per_q": per_q}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "truncated_excess_vs_boiler_tokens"}, indent=1))
    ex = summary["truncated_excess_vs_boiler_tokens"]
    if ex:
        ratios = sorted(b / e for e, b in ex.values() if e > 0)
        print(f"[boiler] truncated: boilerplate tokens / tokens over budget, median {ratios[len(ratios) // 2]:.2f}, "
              f"min {ratios[0]:.2f}, max {ratios[-1]:.2f} (>=1 means removal alone avoids truncation)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["measure"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run dirs with prompt rows")
    ap.add_argument("--contexts", required=True, help="pre-truncation contexts dir (<dom>_contexts.jsonl)")
    ap.add_argument("--ids-file", default=None)
    a = ap.parse_args()
    ids = set(Path(a.ids_file).expanduser().read_text().strip().split(",")) if a.ids_file else None
    measure(a.name, a.runs.split(","), Path(a.contexts).expanduser(), ids)


if __name__ == "__main__":
    main()
