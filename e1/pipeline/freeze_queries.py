#!/usr/bin/env python3
"""Freeze the controller's retrieval queries from the AgentRunbook-R baseline run (E2).

The baseline run stores no query bundle (memory_post_query_metadata is null and vLLM did not log
requests), so the queries are parsed back from each question's rendered memory context:
  raw-state queries : "- Source raw query: ..." lines of the raw-state blocks, in first-seen order
  event query       : "- Event query: ..." header of the event section
  note query        : "- Note query: ..." header of the note section
Caveat: a raw query whose top-6 entries were all taken by earlier queries contributes no block
after the official merge (_merge_raw_state_results), so it is invisible here. It is partly
detectable: the first query always contributes exactly the per-query limit L (6 // n for n <= 3,
2 for n > 3), so L = 6 -> n = 1, L = 3 -> n = 2, L = 2 -> n >= 3. If fewer distinct queries are
seen than L implies, at least one query is missing. Because L = 2 for every n >= 3, the merge rule
is reproduced exactly by passing n_eff = max(#seen, n implied by L) to the merge.

Writes ~/e1/cache/queries/<domain>.jsonl (one row per question) with the queries, the inferred
merge parameters and the baseline raw blocks (query, rank, score, entry id) for the V gate.
Usage: python -I e1/pipeline/freeze_queries.py --runs ~/e1/runs --arm agentrunbook_r_small --out ~/e1/cache/queries
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.common.manifest import write_manifest  # noqa: E402

NOTE_TITLE = "## Procedure and Hint Notes Learned from Previous Tasks in the Environment\n\n"
EVENT_TITLE = "## Retrieved State-Transition Events from Previous Tasks in the Environment\n\n"
RAW_TITLE = "## Retrieved Raw State Slices from Previous Tasks in the Environment\n\n"
RAW_BLOCK = re.compile(
    r"^### Raw state result (\d+)\n- Source raw query: (.*)\n- Retrieval rank within query: (\d+)\n"
    r"- Similarity: (-?[0-9.]+)\n- Trajectory: (\S+)\n- Goal: .*?\n- Center state index: (\d+)\n", re.S)
L_TO_N = {6: 1, 3: 2, 2: 3}


def parse_context(ctx: list[dict]) -> dict:
    note_q = event_q = None
    blocks = []
    for it in ctx:
        if it.get("type") != "text":
            continue
        v = it["value"]
        if v.startswith(NOTE_TITLE):
            m = re.match(r"- Note query: (.*?)\n", v[len(NOTE_TITLE):])
            note_q = m.group(1) if m else None
        elif v.startswith(EVENT_TITLE):
            m = re.match(r"- Event query: (.*?)\n", v[len(EVENT_TITLE):])
            event_q = m.group(1) if m else None
        elif v.startswith("### Raw state result "):
            m = RAW_BLOCK.match(v)
            if not m:
                raise ValueError("unparsed raw block: " + v[:300])
            disp, q, rank, score, tid, center = m.groups()
            blocks.append({"display_rank": int(disp), "query": q, "rank": int(rank), "score": float(score),
                           "entry_id": f"{tid}:raw_state:{int(center):04d}"})
    queries: list[str] = []
    for b in blocks:
        if b["query"] not in queries:
            queries.append(b["query"])
    per_q = Counter(b["query"] for b in blocks)
    L = per_q[queries[0]] if queries else 0
    n_from_L = L_TO_N.get(L)
    n_eff = max(len(queries), n_from_L or 0)
    return {"note_query": note_q, "event_query": event_q, "raw_state_queries": queries,
            "per_query_counts": [per_q[q] for q in queries], "first_query_limit": L,
            "n_queries_seen": len(queries), "n_queries_eff": n_eff,
            "missing_detected": bool(n_from_L and n_from_L > len(queries)),
            "limit_ambiguous": L not in L_TO_N, "baseline_raw_blocks": blocks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="~/e1/runs")
    ap.add_argument("--arm", default="agentrunbook_r_small")
    ap.add_argument("--out", default="~/e1/cache/queries")
    a = ap.parse_args()
    runs, out = Path(a.runs).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for dom in ["web", "enterprise"]:
        src = runs / a.arm / f"{dom}_small" / "per_question.jsonl"
        rows = []
        for r in iter_jsonl(src):
            p = parse_context(r["memory_context"])
            p.update({"question_id": r["question_id"], "domain": dom,
                      "context_truncated": bool(r.get("memory_context_was_truncated"))})
            rows.append(p)
            del r
        dst = out / f"{dom}.jsonl"
        if dst.exists():
            sys.exit(f"{dst} exists; frozen files are write-once")
        with open(dst, "w") as fp:
            for p in rows:
                fp.write(json.dumps(p, ensure_ascii=True) + "\n")
        c = Counter(p["n_queries_seen"] for p in rows)
        print(f"{dom}: {len(rows)} questions -> {dst}")
        print("  raw queries seen per question: " + ", ".join(f"{k}: {c[k]}" for k in range(0, 6)))
        print("  first-query limit L: " + str(dict(sorted(Counter(p['first_query_limit'] for p in rows).items()))))
        print(f"  missing query detected (L implies more queries than seen): {sum(p['missing_detected'] for p in rows)}")
        print(f"  L=2 with <5 queries seen (n in 3..5, a missing query cannot be ruled out): "
              f"{sum(p['first_query_limit'] == 2 and p['n_queries_seen'] < 5 for p in rows)}")
        print(f"  limit ambiguous (L not in 6/3/2): {sum(p['limit_ambiguous'] for p in rows)}")
        print(f"  no event query: {sum(p['event_query'] is None for p in rows)}; no note query: {sum(p['note_query'] is None for p in rows)}")
        print(f"  context truncated at 200k: {sum(p['context_truncated'] for p in rows)}")
        print("  raw blocks per question: " + str(dict(sorted(Counter(len(p['baseline_raw_blocks']) for p in rows).items()))))
    write_manifest(out, inputs={"runs": runs, "arm": a.arm}, params={})


if __name__ == "__main__":
    main()
