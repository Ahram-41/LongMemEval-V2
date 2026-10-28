#!/usr/bin/env python3
"""Per-scenario dense retrieval over turns (index built per scenario before any query: query-blind write).

  bge   : BAAI/bge-base-en-v1.5 (~/e1/models, rev a5beb1e3) with plain transformers, CLS pooling + L2 norm,
          max 512 tokens, fp32 = sentence-transformers' recipe for this model (mem-test v2 retriever).
          Query prefix "Represent this sentence for searching relevant passages: ". Run with the vllm env (torch).
  qwen3 : Qwen3-Embedding-8B through LME_EMBEDDING_BASE_URL (serve_vllm.sh, NO_READER=1); query format
          "Instruct: {task}\\nQuery:{q}". Not the v2 retriever; for later arms.

Writes RET/<retriever>/<set>.jsonl: {uid, order: {dim: [unit index, ...] top TOPN by cosine}}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.manifest import write_manifest  # noqa: E402
from e1.stale.common import DIMS, E1, RET, fmt, load_scenarios  # noqa: E402

BGE = E1 / "models/bge-base-en-v1.5"
BGE_Q = "Represent this sentence for searching relevant passages: "
QWEN_TASK = "Given a question from a user to their personal assistant, retrieve past conversation turns that help answer it."
TOPN = 100


class Bge:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(BGE)
        self.m = AutoModel.from_pretrained(BGE, torch_dtype=torch.float32).to(self.dev).eval()

    def encode(self, texts, batch=64):
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.dev)
                cls = self.m(**enc).last_hidden_state[:, 0]
                out.append(self.torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    def query(self, q):
        return self.encode([BGE_Q + q])[0]


class Qwen3:
    def __init__(self):
        import os
        from openai import OpenAI
        self.cl = OpenAI(base_url=os.environ["LME_EMBEDDING_BASE_URL"], api_key="EMPTY", timeout=600)

    def encode(self, texts, batch=32):
        out = []
        for i in range(0, len(texts), batch):
            r = self.cl.embeddings.create(model="Qwen/Qwen3-Embedding-8B", input=texts[i:i + batch])
            a = np.asarray([d.embedding for d in r.data], dtype=np.float32)
            out.append(a / np.linalg.norm(a, axis=1, keepdims=True))
        return np.concatenate(out)

    def query(self, q):
        return self.encode([f"Instruct: {QWEN_TASK}\nQuery:{q}"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="t1_69")
    ap.add_argument("--retriever", default="bge", choices=["bge", "qwen3"])
    a = ap.parse_args()
    dst = RET / a.retriever
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / f"{a.set}.jsonl"
    done = {json.loads(l)["uid"] for l in open(out)} if out.exists() else set()
    enc = Bge() if a.retriever == "bge" else Qwen3()
    t0 = time.time()
    with open(out, "a") as f:
        for n, sc in enumerate(load_scenarios(a.set), 1):
            if sc["uid"] in done:
                continue
            D = enc.encode([fmt(u) for u in sc["units"]])
            order = {}
            for dim in DIMS:
                s = D @ enc.query(sc["probing_queries"][dim])
                order[dim] = [int(j) for j in np.argsort(-s, kind="stable")[:TOPN]]
            f.write(json.dumps({"uid": sc["uid"], "order": order}) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"[stale retrieve] {a.set}/{a.retriever} {n} scenarios {time.time() - t0:.0f}s", flush=True)
    write_manifest(dst, inputs={"prep": a.set}, params={"retriever": a.retriever, "topn": TOPN}, name=f"manifest_{a.set}.json")
    print(f"[stale retrieve] done {out}", flush=True)


if __name__ == "__main__":
    main()
