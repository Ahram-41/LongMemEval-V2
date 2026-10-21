#!/usr/bin/env python3
"""E2 GPU steps, all through the official AgentRunbook-R object (common/official.py).

  regen   : for questions whose parsed queries are incomplete (stored context truncated at 200k,
            or merge-implied missing query), regenerate the query bundle ONCE with the official
            controller call (_generate_structured_query_bundle: same prompt, Qwen3.5-9B, thinking on,
            temperature 0.6) -> ~/e1/cache/queries_regen/<dom>.jsonl (write-once, resumable)
  queries : embed every query string with the official query path (_embed_texts(is_query=True):
            "Instruct: ...\\nQuery:..." + official truncation) -> ~/e1/index_ext/queries/<dom>/
  pieces  : embed V-full pieces (documents: no instruction, like the official raw-state embedding)
            -> ~/e1/index_ext/dense_full/<dom>/shards/*.npy, then vectors.npy (row = piece_id)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.common.manifest import write_manifest  # noqa: E402
from e1.common.official import E1, load_official  # noqa: E402


def log(m: str) -> None:
    print(f"[e2_gpu {time.strftime('%H:%M:%S')}] {m}", flush=True)


def needs_regen(r: dict) -> bool:
    return bool(r["context_truncated"] or r["missing_detected"])


def do_regen(dom: str, workers: int, limit: int | None) -> None:
    mem = load_official(dom)
    meta = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/meta/{dom}.jsonl")}
    todo = [r["question_id"] for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl") if needs_regen(r)]
    out = E1 / "cache/queries_regen"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"{dom}.jsonl"
    done = {r["question_id"] for r in iter_jsonl(path)} if path.exists() else set()
    todo = [q for q in todo if q not in done][:limit]
    log(f"regen {dom}: {len(done)} cached, {len(todo)} to run")

    def one(qid):
        t0 = time.perf_counter()
        b = mem._generate_structured_query_bundle(query=meta[qid]["question_text"], query_image=None)
        return {"question_id": qid, "query_source": "regenerated", "raw_state_queries": b["raw_state_queries"],
                "event_query": b["event_query"], "note_query": b["note_query"], "seconds": time.perf_counter() - t0}
    with open(path, "a") as fp, ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(one, todo), 1):
            fp.write(json.dumps(row, ensure_ascii=True) + "\n"); fp.flush()
            if i % 20 == 0:
                log(f"regen {dom}: {i}/{len(todo)}")
    write_manifest(out, inputs={"queries": E1 / f"cache/queries/{dom}.jsonl"}, params={"rule": "context_truncated or missing_detected"},
                   name=f"manifest_{dom}.json")


def all_query_strings(dom: str) -> list[dict]:
    rows, seen = [], set()
    def add(src, q):
        if q and (src, q) not in seen:
            seen.add((src, q)); rows.append({"source": src, "text": q})
    for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl"):
        for q in r["raw_state_queries"]:
            add("raw", q)
    p = E1 / f"cache/queries_regen/{dom}.jsonl"
    if p.exists():
        for r in iter_jsonl(p):
            for q in r["raw_state_queries"]:
                add("raw", q)
        for r in iter_jsonl(p):  # regenerated questions get fresh event / note retrieval too
            add("event", r["event_query"])
            add("note", r["note_query"])
    for r in iter_jsonl(E1 / f"cache/meta/{dom}.jsonl"):
        add("question", r["question_text"])
    return rows


def do_queries(dom: str, batch: int) -> None:
    """Incremental: embeds only strings not yet in queries.jsonl / vectors.npy (rows are append-only)."""
    mem = load_official(dom)
    out = E1 / "index_ext/queries" / dom; out.mkdir(parents=True, exist_ok=True)
    have = [r for r in iter_jsonl(out / "queries.jsonl")] if (out / "queries.jsonl").exists() else []
    V_old = np.load(out / "vectors.npy") if have else np.zeros((0, 4096), np.float32)
    assert len(have) == V_old.shape[0]
    known = {(r["source"], r["text"]) for r in have}
    new = [r for r in all_query_strings(dom) if (r["source"], r["text"]) not in known]
    vecs = [V_old]
    for i in range(0, len(new), batch):
        vecs.append(mem._embed_texts([r["text"] for r in new[i:i + batch]], is_query=True).astype(np.float32))
    V = np.concatenate(vecs).astype(np.float32)
    np.save(out / "vectors.tmp.npy", V)
    with open(out / "queries.tmp.jsonl", "w") as fp:
        for i, r in enumerate(have + new):
            fp.write(json.dumps({"row": i, "source": r["source"], "text": r["text"]}, ensure_ascii=True) + "\n")
    (out / "vectors.tmp.npy").rename(out / "vectors.npy"); (out / "queries.tmp.jsonl").rename(out / "queries.jsonl")
    write_manifest(out, inputs={"memory_state": "official"}, params={"format": "official _embed_texts(is_query=True)"})
    log(f"queries {dom}: +{len(new)} -> {V.shape} in {out}")


def do_pieces(dom: str, pieces_name: str, batch: int, workers: int) -> None:
    mem = load_official(dom)
    from memory_modules.agentrunbook_r import _normalize_embeddings
    client, model = mem._get_embedding_client(), mem.embedding_model
    src = E1 / "index_ext/dense_full" / dom / pieces_name
    sh = E1 / "index_ext/dense_full" / dom / ("shards" if pieces_name == "pieces.jsonl" else "shards_" + pieces_name.split(".")[0])
    sh.mkdir(parents=True, exist_ok=True)
    for _ in range(60):  # pieces file is complete only once e2_prep wrote its manifest
        if (src.parent / f"manifest_{pieces_name}.json").exists():
            break
        log(f"waiting for {src} to be completed by e2_prep"); time.sleep(60)
    texts = [r["text"] for r in iter_jsonl(src)]
    n_shard = (len(texts) + batch - 1) // batch
    todo = [k for k in range(n_shard) if not (sh / f"{k:05d}.npy").exists()]
    log(f"pieces {dom}: {len(texts)} pieces, {n_shard} shards, {len(todo)} to embed")
    t0 = time.perf_counter()

    def one(k):
        resp = client.embeddings.create(model=model, input=texts[k * batch:(k + 1) * batch])
        np.save(sh / f"{k:05d}.npy", _normalize_embeddings([d.embedding for d in resp.data]).astype(np.float32))
        return k
    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 20 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                log(f"pieces {dom}: {i}/{len(todo)} shards in {el:.0f}s (eta {el / i * (len(todo) - i):.0f}s)")
    V = np.concatenate([np.load(sh / f"{k:05d}.npy") for k in range(n_shard)])
    assert V.shape[0] == len(texts)
    np.save(sh.parent / ("vectors.npy" if pieces_name == "pieces.jsonl" else f"vectors_{pieces_name.split('.')[0]}.npy"), V)
    write_manifest(sh.parent, inputs={"pieces": src}, params={"batch": batch}, name=f"manifest_vectors_{pieces_name}.json")
    log(f"pieces {dom}: done {V.shape} in {time.perf_counter() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["regen", "queries", "pieces"])
    ap.add_argument("--domains", default="web,enterprise")
    ap.add_argument("--pieces-name", default="pieces.jsonl")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for dom in a.domains.split(","):
        if a.what == "regen":
            do_regen(dom, a.workers, a.limit)
        elif a.what == "queries":
            do_queries(dom, a.batch)
        else:
            do_pieces(dom, a.pieces_name, a.batch, a.workers)


if __name__ == "__main__":
    main()
