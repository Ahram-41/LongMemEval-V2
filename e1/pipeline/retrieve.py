#!/usr/bin/env python3
"""E2 retrieval: rank the raw-state pool per frozen query with a method, then select raw blocks
with the OFFICIAL merge rule (AgentRunbookR._merge_raw_state_results, called unbound).
Events and notes are not touched here (build_context copies or re-searches them).

Two steps, so iterating over method variants is cheap:
  score : for every query string of a domain (frozen raw queries + question texts), compute the
          score of every pool entry for each base method and cache it:
          ~/e1/e2work/scores/<dom>/<method>.npy   (rows follow ~/e1/e2work/scores/<dom>/queries.json)
          base methods: V, Vfull, T:<tok>, G:<src>:<tok>
  select: build raw blocks for one variant -> retrieved.jsonl (one row per question)
          variant = method spec over cached scores: V | Vfull | T:basic | G:both:basic |
                    rrf:<k>:<m1>+<m2>[+...] | w:<w>:<m_dense>+<m_lex>
          query source: frozen | question | both (frozen queries, then the question text)

Frozen queries: ~/e1/cache/queries/<dom>.jsonl (parsed from the B0 contexts); for the questions
whose parsed queries are incomplete, ~/e1/cache/queries_regen/<dom>.jsonl (query_source=regenerated).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.common.official import _stub_transformers  # noqa: E402
from e1.retrieval.base import E1, EntryStore, Query, QueryVectors, official_topk  # noqa: E402
from e1.retrieval.fuse import minmax, ranks  # noqa: E402

TOP_K = 6  # raw_state_search_top_k_per_query
MERGE_CFG = SimpleNamespace(raw_state_result_merge_budget=6, raw_state_result_merge_per_query_cap=2)
WORK = Path(os.environ.get("E2WORK", str(E1 / "e2work")))  # override for scratch runs (e.g. the V gate)


def official_merge(blocks: list[dict]) -> list[dict]:
    _stub_transformers()
    if str(Path(__file__).resolve().parents[2]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import memory_modules.memory  # noqa: F401  (package init order: avoids a circular import)
    from memory_modules.agentrunbook_r import AgentRunbookR
    return AgentRunbookR._merge_raw_state_results(MERGE_CFG, blocks)


def frozen_bundles(dom: str) -> dict[str, dict]:
    """question_id -> {raw_state_queries, event_query, note_query, query_source}."""
    out = {}
    for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl"):
        out[r["question_id"]] = {"raw_state_queries": r["raw_state_queries"], "event_query": r["event_query"],
                                 "note_query": r["note_query"], "query_source": "parsed",
                                 "complete": not (r["context_truncated"] or r["missing_detected"])}
    p = E1 / f"cache/queries_regen/{dom}.jsonl"
    if p.exists():
        for r in iter_jsonl(p):
            out[r["question_id"]].update({"raw_state_queries": r["raw_state_queries"], "event_query": r["event_query"],
                                          "note_query": r["note_query"], "query_source": "regenerated"})
    return out


def meta(dom: str) -> dict[str, dict]:
    return {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/meta/{dom}.jsonl")}


# ---------------------------------------------------------------- score
def query_strings(dom: str) -> list[dict]:
    rows, seen = [], set()
    M = meta(dom)
    for qid, b in frozen_bundles(dom).items():
        for q in b["raw_state_queries"]:
            if q not in seen:
                seen.add(q); rows.append({"text": q, "question_id": qid})
        qt = M[qid]["question_text"]
        if qt not in seen:
            seen.add(qt); rows.append({"text": qt, "question_id": qid})
    return rows


def build_retriever(spec: str, dom: str, store: EntryStore, cache: dict):
    from e1.retrieval.bm25 import BM25
    parts = spec.split(":")
    if parts[0] == "V":
        from e1.retrieval.dense import Dense
        return Dense(dom)
    if parts[0] == "Vfull":
        from e1.retrieval.dense_full import DenseFull
        return DenseFull(dom, store)
    if parts[0] == "Vfulldedup":
        from e1.retrieval.dense_full import DenseFullDedup
        return DenseFullDedup(dom, store)
    if parts[0] == "T":
        key = ("T", parts[1])
        if key not in cache:
            cache[key] = BM25(dom, store, tokenizer=parts[1])
        return cache[key]
    if parts[0] == "G":
        from e1.retrieval.grep import Grep
        key = ("T", parts[2])
        if key not in cache:
            cache[key] = BM25(dom, store, tokenizer=parts[2])
        return Grep(store, cache[key], source=parts[1])
    raise ValueError(spec)


def do_score(dom: str, methods: list[str], force: bool = False) -> None:
    d = WORK / "scores" / dom
    d.mkdir(parents=True, exist_ok=True)
    rows = query_strings(dom)
    qfile = d / "queries.json"
    if qfile.exists():
        old = json.loads(qfile.read_text())
        assert [r["text"] for r in old] == [r["text"] for r in rows], "query set changed; use a new work dir"
    else:
        qfile.write_text(json.dumps(rows))
    todo = [m for m in methods if force or not (d / f"{m.replace(':', '_')}.npy").exists()]
    if not todo:
        return
    need_text = any(m.startswith(("T", "G")) for m in todo)
    store = EntryStore.load(dom, with_text=need_text)
    M = meta(dom)
    qv = QueryVectors(dom) if any(m in ("V", "Vfull", "Vfulldedup") for m in todo) else None
    cache: dict = {}
    for m in todo:
        r = build_retriever(m, dom, store, cache)
        S = np.zeros((len(rows), len(store)), dtype=np.float32)
        for i, row in enumerate(rows):
            q = Query(text=row["text"], question_id=row["question_id"], question_text=M[row["question_id"]]["question_text"],
                      vec=qv.get(row["text"]) if qv else None)
            S[i] = r.scores(q)
        np.save(d / f"{m.replace(':', '_')}.npy", S)
        print(f"[score] {dom} {m}: {S.shape}", flush=True)


def base_methods(variant: str) -> list[str]:
    v = variant.split("@")[0]
    if v.startswith(("rrf:", "w:")):
        return v.split(":", 2)[2].split("+")
    return [v]


# ---------------------------------------------------------------- select
class ScoreBank:
    def __init__(self, dom: str):
        self.d = WORK / "scores" / dom
        self.rows = json.loads((self.d / "queries.json").read_text())
        self.row = {r["text"]: i for i, r in enumerate(self.rows)}
        self.mats: dict[str, np.ndarray] = {}

    def mat(self, m: str) -> np.ndarray:
        if m not in self.mats:
            self.mats[m] = np.load(self.d / f"{m.replace(':', '_')}.npy", mmap_mode="r")
        return self.mats[m]

    def scores(self, variant: str, text: str) -> np.ndarray:
        i = self.row[text]
        if variant.startswith("rrf:"):
            _, k, members = variant.split(":", 2)
            return sum(1.0 / (float(k) + ranks(np.asarray(self.mat(m)[i], dtype=np.float64))) for m in members.split("+"))
        if variant.startswith("w:"):
            _, w, members = variant.split(":", 2)
            a, b = members.split("+")
            w = float(w)
            return (1 - w) * minmax(np.asarray(self.mat(a)[i], np.float64)) + w * minmax(np.asarray(self.mat(b)[i], np.float64))
        return np.asarray(self.mat(variant)[i], dtype=np.float64)


def queries_for(bundle: dict, question_text: str, source: str) -> list[str]:
    if source == "frozen":
        return list(bundle["raw_state_queries"])
    if source == "question":
        return [question_text]
    if source == "both":
        return list(bundle["raw_state_queries"]) + ([question_text] if question_text not in bundle["raw_state_queries"] else [])
    raise ValueError(source)


def select_question(bank: ScoreBank, store_ids: list[str], variant: str, queries: list[str], vbank_variant: str | None = "V"):
    """Official search (top 6 per query, official ordering) + official merge. The rendered
    'Similarity' of each block is the official V cosine (stored embedding x query vector) so the
    reader-visible header keeps its meaning; 'rank' is the method's rank within the query."""
    blocks = []
    for q in queries:
        s = bank.scores(variant, q)
        top = official_topk(s, TOP_K)
        sim = bank.scores(vbank_variant, q) if vbank_variant else s
        blocks.append({"query": q, "results": [{"query": q, "rank": r, "score": float(sim[i]), "method_score": float(s[i]),
                                                 "entry": {"entry_id": store_ids[i]}} for r, i in enumerate(top, start=1)]})
    merged = official_merge(blocks)
    return [{"query": m["query"], "rank": m["rank"], "score": m["score"], "method_score": m["method_score"],
             "entry_id": m["entry"]["entry_id"]} for m in merged]


def entry_ids(dom: str) -> list[str]:
    ids = []
    with open(E1 / f"index/agentrunbook_r_{dom}_small/memory_state/pools/raw_state_pool.jsonl") as fp:
        for line in fp:
            ids.append(line[line.index('"entry_id": "') + 13:].split('"', 1)[0])
    return ids


def do_select(dom: str, variant: str, source: str, out: Path, bank: ScoreBank | None = None, ids=None,
              sim_variant: str | None = "V") -> list[dict]:
    bank = bank or ScoreBank(dom)
    ids = ids or entry_ids(dom)
    B, M = frozen_bundles(dom), meta(dom)
    rows = []
    for qid in sorted(B):
        b = B[qid]
        qs = queries_for(b, M[qid]["question_text"], source)
        rows.append({"question_id": qid, "variant": variant, "query_source_mode": source, "query_source": b["query_source"],
                     "complete": b["complete"], "queries": qs, "raw_blocks": select_question(bank, ids, variant, qs, sim_variant)})
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=True) + "\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["score", "select"])
    ap.add_argument("--domains", default="web")
    ap.add_argument("--methods", default="V,Vfull,T:basic,G:both:basic")
    ap.add_argument("--variant", default="V")
    ap.add_argument("--source", default="frozen")
    ap.add_argument("--out", default=None, help="select: output path (default ~/e1/e2work/retrieved/<variant>__<source>/<dom>.jsonl)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for dom in a.domains.split(","):
        if a.what == "score":
            do_score(dom, a.methods.split(","), a.force)
        else:
            out = Path(a.out).expanduser() if a.out else WORK / "retrieved" / f"{a.variant.replace(':', '_')}__{a.source}" / f"{dom}.jsonl"
            do_select(dom, a.variant, a.source, out)
            print(f"[select] {dom} {a.variant} {a.source} -> {out}")


if __name__ == "__main__":
    main()
