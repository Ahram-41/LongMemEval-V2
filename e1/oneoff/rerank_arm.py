"""Cross-encoder reranking arm (Qwen3-Reranker-8B) on top of the E2 first-stage retrievers.

  pairs   per rule-scored question and frozen query: candidate entries = union of the top-N under
          V, Vfull and T:camel (score matrices from retrieve.py score); the document shown to the
          reranker is the entry's best piece (highest Qwen3-Embedding cosine to that query among the
          <=4,096-token pieces of index_ext/dense_full), cut to MAX_CHARS. Writes rerank/pairs_<dom>.jsonl.
  select  reads rerank/scores_<dom>.jsonl (from rerank_score.py), takes the top-6 per query by
          reranker P(yes), official merge, writes retrieved/rr/<dom>.jsonl (+ ids.txt of changed questions).
          The rendered 'Similarity' stays the official V cosine so the reader-visible header keeps its meaning.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.pipeline.retrieve import E1, TOP_K, WORK, ScoreBank, entry_ids, official_merge  # noqa: E402
from e1.retrieval.base import QueryVectors, official_topk  # noqa: E402

DOMS = ("web", "enterprise")
MEMBERS = ("V", "Vfull", "T:camel")
TOP_N = 15
MAX_CHARS = 9000  # ~2.5k tokens of AXTree text
OUT = WORK / "rerank"


def rule_scored(dom: str) -> set[str]:
    return {r["question_id"] for r in iter_jsonl(E1 / f"cache/meta/{dom}.jsonl")
            if not r["is_abstention_problem"] and not r["eval_function"].startswith("llm_")}


def pieces_by_entry(dom: str):
    by = defaultdict(list)
    texts = {}
    for p in iter_jsonl(E1 / f"index_ext/dense_full/{dom}/pieces.jsonl"):
        eid = f"{p['trajectory_id']}:raw_state:{p['state_index']:04d}"
        by[eid].append(p["piece_id"])
        texts[p["piece_id"]] = p["text"]
    vec = np.load(E1 / f"index_ext/dense_full/{dom}/vectors.npy", mmap_mode="r")
    return by, texts, vec


def pairs(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for dom in DOMS:
        keep = rule_scored(dom)
        bank, store = ScoreBank(dom), entry_ids(dom)
        by, texts, vec = pieces_by_entry(dom)
        qv = QueryVectors(dom)
        n_pairs, n_q = 0, 0
        with open(OUT / f"pairs_{dom}.jsonl", "w") as fp:
            for r in iter_jsonl(WORK / f"retrieved/e2_v/{dom}.jsonl"):
                if r["question_id"] not in keep:
                    continue
                for q in r["queries"]:
                    cand = []
                    for m in MEMBERS:
                        cand.extend(official_topk(bank.scores(m, q), TOP_N))
                    cand = sorted(set(cand))
                    v = qv.get(q)
                    n_q += 1
                    for i in cand:
                        eid = store[i]
                        pids = by[eid]
                        sims = np.asarray(vec[pids]) @ v
                        best = pids[int(np.argmax(sims))]
                        fp.write(json.dumps({"question_id": r["question_id"], "query": q, "entry_id": eid,
                                             "piece_id": best, "piece_sim": float(sims.max()),
                                             "v": float(bank.scores("V", q)[i]), "doc": texts[best][:MAX_CHARS]},
                                            ensure_ascii=True) + "\n")
                        n_pairs += 1
        print(f"[pairs] {dom}: {n_q} queries, {n_pairs} pairs -> {OUT / f'pairs_{dom}.jsonl'}", flush=True)


def select(args) -> None:
    ids_changed = []
    for dom in DOMS:
        sc = {}
        for s in iter_jsonl(OUT / f"scores_{dom}.jsonl"):
            sc[(s["question_id"], s["query"], s["entry_id"])] = s["p_yes"]
        vs = {(p["question_id"], p["query"], p["entry_id"]): p["v"] for p in iter_jsonl(OUT / f"pairs_{dom}.jsonl")}
        byq = defaultdict(list)
        for (qid, q, eid), p in sc.items():
            byq[(qid, q)].append((p, eid))
        out = WORK / "retrieved/rr" / f"{dom}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out, "w") as fp:
            for r in iter_jsonl(WORK / f"retrieved/e2_v/{dom}.jsonl"):
                qid = r["question_id"]
                if r["queries"] and (qid, r["queries"][0]) in byq:
                    blocks = []
                    for q in r["queries"]:
                        top = sorted(byq[(qid, q)], reverse=True)[:TOP_K]
                        blocks.append({"query": q, "results": [
                            {"query": q, "rank": k, "score": vs[(qid, q, eid)], "method_score": p, "entry": {"entry_id": eid}}
                            for k, (p, eid) in enumerate(top, start=1)]})
                    merged = official_merge(blocks)
                    new = [{"query": m["query"], "rank": m["rank"], "score": m["score"], "method_score": m["method_score"],
                            "entry_id": m["entry"]["entry_id"]} for m in merged]
                    if [b["entry_id"] for b in new] != [b["entry_id"] for b in r["raw_blocks"]]:
                        ids_changed.append(qid)
                        n += 1
                    r = {**r, "variant": "rr:Qwen3-Reranker-8B", "raw_blocks": new}
                fp.write(json.dumps(r, ensure_ascii=True) + "\n")
        print(f"[select] {dom}: {n} questions with a different block list -> {out}", flush=True)
    (OUT / "ids.txt").write_text(",".join(sorted(ids_changed)))
    print(f"[select] changed total {len(ids_changed)}")


def matrix(args) -> None:
    """E2-format score matrix of the reranker (rows = e2work/scores/<dom>/queries.json, cols = pool order;
    non-candidates = -inf) -> ~/e1/v2work/scores/<dom>/RR.npy, so step_keys.py eval can score coverage
    on the evidence rules and the guardrails without a reader run."""
    for dom in DOMS:
        rows = json.loads((WORK / "scores" / dom / "queries.json").read_text())
        row = {r["text"]: i for i, r in enumerate(rows)}
        store = entry_ids(dom)
        pos = {e: i for i, e in enumerate(store)}
        M = np.full((len(rows), len(store)), -np.inf, dtype=np.float32)
        n = 0
        for s in iter_jsonl(OUT / f"scores_{dom}.jsonl"):
            M[row[s["query"]], pos[s["entry_id"]]] = s["p_yes"]
            n += 1
        out = E1 / "v2work/scores" / dom / "RR.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, M)
        print(f"[matrix] {dom}: {n} scored cells in {M.shape} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pairs", "select", "matrix"])
    args = ap.parse_args()
    {"pairs": pairs, "select": select, "matrix": matrix}[args.what](args)


if __name__ == "__main__":
    main()
