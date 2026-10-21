#!/usr/bin/env python3
"""Render E2 memory contexts with the official AgentRunbookR renderers (order: notes, events, raw).

Raw-state section: official _build_raw_state_context_items on the selected blocks
(retrieve.py output; 'Similarity' = official V cosine, 'rank' = the method's rank in its query).
Notes and events:
  parsed questions (queries recovered from an untruncated B0 context): copied item-for-item from
      the B0 memory context (everything before the raw-state section title)
  regenerated questions: official search (_search_entries semantics: stored embeddings x official
      query vector, np.argsort(...)[::-1][:top_k]) with the regenerated event / note query, then the
      official _build_note_context_items / _build_event_context_items
Gate mode (--gate): render the B0 raw blocks parsed from the B0 context (cache/queries) and require
the result to equal the B0 memory context exactly, for every untruncated question.

Runs on a CPU node (transformers stubbed; the renderers never touch the tokenizer).
Usage: python -I e1/pipeline/build_context.py --retrieved <dir>/<dom>.jsonl --out <dir>/<dom>_contexts.jsonl --domain web
       python -I e1/pipeline/build_context.py --gate --domain web
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.common.official import E1, load_official  # noqa: E402
from e1.retrieval.base import QueryVectors, official_topk  # noqa: E402

RAW_TITLE = "## Retrieved Raw State Slices from Previous Tasks in the Environment\n\n"


def b0_prefix_items(ctx: list[dict]) -> list[dict]:
    out = []
    for it in ctx:
        if it.get("type") == "text" and it["value"] == RAW_TITLE:
            break
        out.append(it)
    return out


class Renderer:
    def __init__(self, dom: str):
        self.mem = load_official(dom, cpu=True)
        self.idx = {e["entry_id"]: i for i, e in enumerate(self.mem.raw_state_entries)}
        self.qv = None
        self.dom = dom

    def raw_items(self, blocks: list[dict]) -> list[dict]:
        results = [{"query": b["query"], "rank": b["rank"], "score": b["score"],
                    "entry": self.mem.raw_state_entries[self.idx[b["entry_id"]]]} for b in blocks]
        return self.mem._build_raw_state_context_items(results)

    def _search(self, entries, emb, text, k):
        if not text.strip() or emb.size == 0 or not entries:
            return []
        if self.qv is None:
            self.qv = QueryVectors(self.dom)
        s = emb @ self.qv.get(text)
        return [{"query": text, "rank": r, "score": float(s[i]), "entry": entries[i]}
                for r, i in enumerate(official_topk(s, k), start=1)]

    def fresh_prefix(self, event_query: str, note_query: str) -> list[dict]:
        m = self.mem
        notes = {"procedure_results": [], "hint_results": []} if not note_query else {
            "procedure_results": self._search(m.procedure_note_entries, m.procedure_note_embeddings, note_query, m.note_search_top_k_per_type),
            "hint_results": self._search(m.hint_note_entries, m.hint_note_embeddings, note_query, m.note_search_top_k_per_type)}
        events = self._search(m.event_entries, m.event_embeddings, event_query, m.event_search_top_k) if event_query else []
        return m._build_note_context_items(note_query, notes) + m._build_event_context_items(event_query, events)


def stream_b0(dom: str):
    for r in iter_jsonl(E1 / f"runs/agentrunbook_r_small/{dom}_small/per_question.jsonl"):
        yield r["question_id"], r["memory_context"], bool(r.get("memory_context_was_truncated"))
        del r


def gate(dom: str) -> None:
    R = Renderer(dom)
    Q = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl")}
    n = same = 0
    bad = []
    for qid, ctx, trunc in stream_b0(dom):
        if trunc:
            continue
        n += 1
        new = b0_prefix_items(ctx) + R.raw_items(Q[qid]["baseline_raw_blocks"])
        if new == ctx:
            same += 1
        else:
            bad.append(qid)
    print(f"[gate] {dom}: rendered B0 blocks reproduce the B0 context exactly for {same}/{n} untruncated questions"
          + (f"; differing: {bad[:10]}" if bad else ""))


def build(dom: str, retrieved: Path, out: Path) -> None:
    R = Renderer(dom)
    rows = {r["question_id"]: r for r in iter_jsonl(retrieved)}
    regen = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/queries_regen/{dom}.jsonl")} \
        if (E1 / f"cache/queries_regen/{dom}.jsonl").exists() else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fp:
        for qid, ctx, trunc in stream_b0(dom):
            r = rows[qid]
            if r["query_source"] == "regenerated":
                g = regen[qid]
                prefix = R.fresh_prefix(g["event_query"], g["note_query"])
            else:
                assert not trunc
                prefix = b0_prefix_items(ctx)
            mc = prefix + R.raw_items(r["raw_blocks"])
            fp.write(json.dumps({"question_id": qid, "memory_context": mc,
                                 "raw_block_ids": [b["entry_id"] for b in r["raw_blocks"]]}, ensure_ascii=True) + "\n")
    print(f"[build] {dom}: {len(rows)} contexts -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--retrieved")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.gate:
        gate(a.domain)
    else:
        build(a.domain, Path(a.retrieved).expanduser(), Path(a.out).expanduser())


if __name__ == "__main__":
    main()
