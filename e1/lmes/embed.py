#!/usr/bin/env python3
"""E3 embeddings through a local vLLM Qwen3-Embedding-8B server (LME_EMBEDDING_BASE_URL, set by
e1/slurm/serve_vllm.sh). Run with the lme-v2-release env python (openai client).

  pieces  : ~/e1/e3work/prep/pieces.jsonl (documents, no instruction) -> ~/e1/e3work/index/dense/shards/*.npy,
            then vectors.npy (row = piece_id). Resumable; --limit N embeds an evenly strided sample of N pieces into
            shards_test/ for timing.
  queries : question text with the Qwen3-Embedding query format "Instruct: {task}\\nQuery:{question}" ->
            ~/e1/e3work/index/dense/queries.{jsonl,npy} (row per question, order of questions.jsonl)

Vectors are L2-normalised (same as the official _normalize_embeddings). Pieces are <= 4,096 tokens and the
server runs with --max-model-len 8192, so nothing is truncated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.common.manifest import write_manifest  # noqa: E402
from e1.lmes.prep import PREP, WORK  # noqa: E402

DENSE = WORK / "index/dense"
MODEL = "Qwen/Qwen3-Embedding-8B"
# Task text written for this dataset (the official LME-V2 one speaks of "agent trajectories"); format is
# Qwen3-Embedding's documented query format. Fixed before any dense result.
TASK = "Given a question about the user's past chat sessions with an AI assistant, retrieve the conversation rounds that help answer it."


def log(m: str) -> None:
    print(f"[e3_embed {time.strftime('%H:%M:%S')}] {m}", flush=True)


def client():
    from openai import OpenAI
    return OpenAI(base_url=os.environ["LME_EMBEDDING_BASE_URL"], api_key="EMPTY", timeout=600)


def norm(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def embed_batches(texts: list[str], sh: Path, batch: int, workers: int) -> np.ndarray:
    cl = client()
    sh.mkdir(parents=True, exist_ok=True)
    n_shard = (len(texts) + batch - 1) // batch
    todo = [k for k in range(n_shard) if not (sh / f"{k:05d}.npy").exists()]
    log(f"{len(texts)} texts, {n_shard} shards, {len(todo)} to embed -> {sh}")
    t0 = time.perf_counter()

    def one(k):
        resp = cl.embeddings.create(model=MODEL, input=texts[k * batch:(k + 1) * batch])
        np.save(sh / f"{k:05d}.npy", norm([d.embedding for d in resp.data]).astype(np.float32))
        return k
    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 50 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                log(f"{i}/{len(todo)} shards in {el:.0f}s (eta {el / i * (len(todo) - i):.0f}s)")
    log(f"embedded in {time.perf_counter() - t0:.1f}s")
    V = np.concatenate([np.load(sh / f"{k:05d}.npy") for k in range(n_shard)])
    assert V.shape[0] == len(texts), (V.shape, len(texts))
    return V


def do_pieces(batch: int, workers: int, limit: int | None) -> None:
    rows = [(r["text"], r["n_tokens"]) for r in iter_jsonl(PREP / "pieces.jsonl")]
    if limit:  # timing sample: every k-th piece (deterministic, spans the whole length distribution)
        rows = rows[::max(1, len(rows) // limit)][:limit]
    texts = [t for t, _ in rows]
    ntok = sum(n for _, n in rows)
    t0 = time.perf_counter()
    V = embed_batches(texts, DENSE / ("shards_test" if limit else "shards"), batch, workers)
    el = time.perf_counter() - t0
    log(f"pieces: {len(texts)} pieces, {ntok} tokens, {el:.1f}s wall incl. reload ({ntok / max(el, 1e-9):.0f} tok/s)")
    name = "vectors_test.npy" if limit else "vectors.npy"
    np.save(DENSE / name, V)
    write_manifest(DENSE, inputs={"pieces": PREP / "pieces.jsonl"},
                   params={"batch": batch, "workers": workers, "limit": limit, "tokens": ntok, "seconds": round(el, 1)},
                   name=f"manifest_{name}.json")


def do_queries(batch: int, workers: int) -> None:
    qs = [(r["question_id"], r["question"]) for r in iter_jsonl(PREP / "questions.jsonl")]
    texts = [f"Instruct: {TASK}\nQuery:{q.strip()}".strip() for _, q in qs]
    V = embed_batches(texts, DENSE / "shards_queries", batch, workers)
    np.save(DENSE / "queries.npy", V)
    with open(DENSE / "queries.jsonl", "w") as fp:
        for i, ((qid, _), t) in enumerate(zip(qs, texts)):
            fp.write(json.dumps({"row": i, "question_id": qid, "text": t}, ensure_ascii=True) + "\n")
    write_manifest(DENSE, inputs={"questions": PREP / "questions.jsonl"}, params={"task": TASK, "format": "Instruct: {task}\\nQuery:{q}"},
                   name="manifest_queries.json")
    log(f"queries: {V.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pieces", "queries"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.what == "pieces":
        do_pieces(a.batch, a.workers, a.limit)
    else:
        do_queries(a.batch, a.workers)


if __name__ == "__main__":
    main()
