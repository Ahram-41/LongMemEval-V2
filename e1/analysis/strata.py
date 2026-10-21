#!/usr/bin/env python3
"""E2 §十一 strata, FIXED BEFORE any method comparison (committed before retrieval_eval is run on
V-full / T / G / H). Every stratum is computed from the pool and the question only, never from a
method's retrieved blocks.

Diagnostic strata (need the gold answer; for explanation only, never for routing). Defined on
normal questions with a phrase answer (official split / normalisation, see retrieval_eval):
  pos    : evidence position w.r.t. the official embedding input. An entry is answer-bearing if
           its full slice_axtree_text (normalised) contains every answer phrase. Its embedded part
           is the official _truncate_for_embedding: middle cut to 40,000 chars, then the first
           4,096 tokens of the Qwen3-Embedding-8B tokenizer.
             front    : some answer-bearing entry has every phrase inside its embedded part
             tail     : answer-bearing entries exist, none has every phrase in its embedded part
             absent   : no raw entry contains every phrase
  plen   : page length of the answer-bearing evidence = token count of the SHORTEST answer-bearing
           entry's slice text; terciles over the eligible questions of the domain (low/mid/high)
  rarity : number of raw entries containing every phrase; fixed bins 0 / 1-5 / 6-50 / >50
Routable strata (question + pool only; usable for routing later):
  qtype  : question_type
  premise: normal / false-premise
  domain : web / enterprise (tables are per domain anyway)
  anchors: number of quoted UI terms in the question: 0 / 1 / 2+
  qidf   : mean BM25 idf (basic tokenizer, raw pool) of the distinct frozen-query tokens of the
           question; terciles over all questions of the domain (low/mid/high)
Writes ~/e1/e2work/strata/<dom>.json  {question_id: {stratum: label}}.
Usage: python -I e1/analysis/strata.py --domains web,enterprise
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.analysis.retrieval_eval import _official_metrics, answer_phrases  # noqa: E402
from e1.common.anchors import anchors  # noqa: E402
from e1.common.io import iter_jsonl  # noqa: E402
from e1.pipeline.retrieve import E1, frozen_bundles, meta  # noqa: E402

TOKENIZER = E1 / "hf_cache/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/tokenizer.json"
CHAR_CAP, TOK_CAP = 40000, 4096  # EMBEDDING_PRETOKEN_CHAR_CAP, embedding max_input_tokens


def truncate_middle(text: str, max_chars: int) -> str:
    """Copy of the official _truncate_middle (memory_modules/agentrunbook_r.py)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 64:
        return text[:max_chars]
    omitted_chars = len(text) - max_chars
    marker = f"\n...[truncated {omitted_chars} chars]...\n"
    if len(marker) >= max_chars:
        return text[:max_chars]
    head_chars = (max_chars - len(marker)) // 2
    tail_chars = max_chars - len(marker) - head_chars
    return f"{text[:head_chars].rstrip()}{marker}{text[-tail_chars:].lstrip()}".strip()


def tercile(values: dict[str, float]) -> dict[str, str]:
    if not values:
        return {}
    a = np.array(list(values.values()), dtype=float)
    q1, q2 = np.quantile(a, [1 / 3, 2 / 3])
    return {k: ("low" if v <= q1 else "mid" if v <= q2 else "high") for k, v in values.items()}


def rarity_bin(n: int) -> str:
    return "0" if n == 0 else "1-5" if n <= 5 else "6-50" if n <= 50 else ">50"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="web,enterprise")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    from e1.retrieval.bm25 import BM25, tokenize
    from e1.retrieval.base import EntryStore
    tok = Tokenizer.from_file(str(TOKENIZER))
    norm = _official_metrics()[0]
    for dom in a.domains.split(","):
        M = meta(dom)
        B = frozen_bundles(dom)
        st = EntryStore.load(dom, with_text=True)
        full_norm, emb_norm, ntok = [], [], []
        for t in st.texts:
            cut = truncate_middle(t, CHAR_CAP)
            enc = tok.encode(cut, add_special_tokens=False)
            emb = cut if len(enc.ids) <= TOK_CAP else cut[: enc.offsets[TOK_CAP - 1][1]]
            full_norm.append(" " + norm(t) + " ")
            emb_norm.append(" " + norm(emb) + " ")
            ntok.append(len(tok.encode(t, add_special_tokens=False).ids))
        bm = BM25(dom, st, tokenizer="basic")
        out: dict[str, dict] = {}
        plen_raw: dict[str, float] = {}
        qidf: dict[str, float] = {}
        for qid, m in M.items():
            s = {"qtype": m["question_type"], "premise": "false-premise" if m["is_abstention_problem"] else "normal",
                 "domain": dom}
            na = len(anchors(m["question_text"]))
            s["anchors"] = "0" if na == 0 else "1" if na == 1 else "2+"
            terms = sorted({w for q in B[qid]["raw_state_queries"] for w in tokenize(q, "basic")})
            qidf[qid] = float(np.mean([bm.idf(w) for w in terms])) if terms else 0.0
            ph = answer_phrases(m)
            if ph:
                bearing = [i for i, t in enumerate(full_norm) if all(f" {p} " in t for p in ph)]
                s["rarity"] = rarity_bin(len(bearing))
                if not bearing:
                    s["pos"] = "absent"
                else:
                    s["pos"] = "front" if any(all(f" {p} " in emb_norm[i] for p in ph) for i in bearing) else "tail"
                    plen_raw[qid] = min(ntok[i] for i in bearing)
            out[qid] = s
        for qid, lab in tercile(plen_raw).items():
            out[qid]["plen"] = lab
        for qid, lab in tercile(qidf).items():
            out[qid]["qidf"] = lab
        d = E1 / "e2work/strata"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{dom}.json").write_text(json.dumps(out, indent=0))
        # per-entry normalised slice text / embedded prefix, reused by the per-phrase hit dump
        import pickle
        (d / f"{dom}_entrytext.pkl").write_bytes(pickle.dumps(
            {"entry_ids": st.entry_ids, "full_norm": full_norm, "emb_norm": emb_norm, "ntok": ntok}, protocol=pickle.HIGHEST_PROTOCOL))
        from collections import Counter
        for k in ["pos", "plen", "rarity", "anchors", "qidf"]:
            print(f"[strata] {dom} {k}: {dict(Counter(v[k] for v in out.values() if k in v))}")


if __name__ == "__main__":
    main()
