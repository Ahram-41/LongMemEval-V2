#!/usr/bin/env python3
"""Base round scores per question (CPU). One float32 array per question over the flat round list of its
haystack (questions.jsonl r_uid order) -> ~/e1/e3work/scores/<method>.npz (key = question_id).

  T_<tok> : BM25 (e1/retrieval/bm25.py formula, k1 = 1.2, b = 0.75) over the rounds of THIS question's
            haystack only (idf and average length from that haystack); query = question text.
  V       : cosine of the question embedding (with instruction) and each round; a round longer than
            4,096 tokens scores the max over its pieces.
Scores are computed for all questions (dev and held-out); evaluation of held-out is gated in evaluate.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.common.manifest import write_manifest  # noqa: E402
from e1.lmes.prep import PREP, WORK  # noqa: E402
from e1.retrieval.bm25 import BM25, tokenize  # noqa: E402

SCORES = WORK / "scores"


def round_texts() -> list[str]:
    out = []
    for r in iter_jsonl(PREP / "rounds.jsonl"):
        assert r["uid"] == len(out)
        out.append(r["text"])
    return out


def do_bm25(tok: str) -> None:
    texts = round_texts()
    out = {}
    t0 = time.perf_counter()
    for q in iter_jsonl(PREP / "questions.jsonl"):
        bm = BM25.from_texts([texts[u] for u in q["r_uid"]], tokenizer=tok)
        out[q["question_id"]] = bm.scores_terms(tokenize(q["question"], tok)).astype(np.float32)
    SCORES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCORES / f"T_{tok}.npz", **out)
    write_manifest(SCORES, inputs={"questions": PREP / "questions.jsonl", "rounds": PREP / "rounds.jsonl"},
                   params={"method": f"T_{tok}", "k1": 1.2, "b": 0.75, "corpus": "per-question haystack rounds"},
                   name=f"manifest_T_{tok}.json")
    print(f"T_{tok}: {len(out)} questions in {time.perf_counter() - t0:.0f}s")


def do_dense() -> None:
    D = WORK / "index/dense"
    P = np.load(D / "vectors.npy", mmap_mode="r")
    uid = np.array([r["uid"] for r in iter_jsonl(PREP / "pieces.jsonl")])
    assert P.shape[0] == len(uid)
    Q = np.load(D / "queries.npy")
    qrow = {r["question_id"]: r["row"] for r in iter_jsonl(D / "queries.jsonl")}
    starts = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
    assert len(starts) == uid.max() + 1 and (uid[starts] == np.arange(len(starts))).all(), "pieces must be contiguous per uid"
    S = np.empty((P.shape[0], Q.shape[0]), dtype=np.float32)   # every piece x every question
    for a in range(0, P.shape[0], 20000):
        S[a:a + 20000] = np.asarray(P[a:a + 20000]) @ Q.T
    R = np.maximum.reduceat(S, starts, axis=0)                  # round score = max over its pieces
    del S
    out = {}
    for q in iter_jsonl(PREP / "questions.jsonl"):
        out[q["question_id"]] = R[np.array(q["r_uid"]), qrow[q["question_id"]]].astype(np.float32)
    SCORES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCORES / "V.npz", **out)
    write_manifest(SCORES, inputs={"pieces": D / "vectors.npy", "queries": D / "queries.npy"},
                   params={"method": "V", "round_score": "max over <=4096-token pieces"}, name="manifest_V.json")
    print(f"V: {len(out)} questions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["bm25", "dense"])
    ap.add_argument("--tok", default="camel")
    a = ap.parse_args()
    do_bm25(a.tok) if a.what == "bm25" else do_dense()


if __name__ == "__main__":
    main()
