#!/usr/bin/env python3
"""A (pre-registered in CP5101 experiments/alloc-plan-2026-11-02.md): count-matched allocation of the
raw-state blocks across a question's queries. Ranking matrices are untouched; only which N_q blocks are
taken changes, with N_q = the block count the official merge gives for the same base ranking.

Variant syntax: <base>~<rule>, rule in
  official : my re-implementation of the official merge (self-test only; must equal retrieve.official_merge)
  consD    : candidates = union of each query's top D; s(e) = sum_q 1 / (60 + rank_q(e)); top N_q by s
  zmaxD    : same candidates; s(e) = max_q z_q(e), z over the whole pool per query (raw-score bases only)
Ties keep the official merge order, then best rank over queries, then query order.

  selftest : official re-implementation vs retrieve.select_question, every question, given bases
  eval     : step_keys eval (coverage on the frozen evidence rules + guardrails) with these variants
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_keys  # noqa: E402  (sets $E2WORK to the v2 work dir before retrieve is imported)
from e1.pipeline import retrieve  # noqa: E402
from e1.retrieval.base import official_topk  # noqa: E402
from e1.retrieval.fuse import ranks  # noqa: E402

RRF_K = 60
_official_select = retrieve.select_question


def official_ids(bank, variant: str, queries: list[str]) -> list[str]:
    """Official merge, re-implemented: per-query limit 6 // n for n <= 3 queries, else the cap of 2."""
    n = len(queries)
    if n == 0:
        return []
    limit = max(1, retrieve.MERGE_CFG.raw_state_result_merge_budget // n) if n <= 3 \
        else retrieve.MERGE_CFG.raw_state_result_merge_per_query_cap
    out, seen = [], set()
    for q in queries:
        taken = 0
        for i in official_topk(bank.scores(variant, q), retrieve.TOP_K):
            if i in seen:
                continue
            seen.add(i); out.append(i); taken += 1
            if taken >= limit:
                break
    return out


def allocate(bank, base: str, rule: str, queries: list[str]) -> list[int]:
    off = official_ids(bank, base, queries)
    if rule == "official" or not queries:
        return off
    m = re.fullmatch(r"(cons|zmax)(\d+)", rule)
    kind, depth = m.group(1), int(m.group(2))
    S = [np.asarray(bank.scores(base, q), dtype=np.float64) for q in queries]
    R = [ranks(s) for s in S]
    cand = set()
    for s in S:
        cand.update(official_topk(s, depth))
    if kind == "cons":
        score = {i: sum(1.0 / (RRF_K + r[i]) for r in R) for i in cand}
    else:
        Z = [(s - s.mean()) / (s.std() or 1.0) for s in S]
        score = {i: max(z[i] for z in Z) for i in cand}
    opos = {i: p for p, i in enumerate(off)}
    best = {i: min(r[i] for r in R) for i in cand}
    qfirst = {i: min(k for k, r in enumerate(R) if r[i] == best[i]) for i in cand}
    order = sorted(cand, key=lambda i: (-score[i], opos.get(i, len(off)), best[i], qfirst[i]))
    return order[:len(off)]


def select_question(bank, store_ids, variant, queries, vbank_variant="V"):
    if "~" not in variant:
        return _official_select(bank, store_ids, variant, queries, vbank_variant)
    base, rule = variant.split("~", 1)
    return [{"query": None, "rank": p, "score": None, "method_score": None, "entry_id": store_ids[i]}
            for p, i in enumerate(allocate(bank, base, rule, queries), start=1)]


def selftest(bases: list[str]) -> None:
    bad = 0
    for dom in step_keys.DOMS:
        bank, ids = retrieve.ScoreBank(dom), retrieve.entry_ids(dom)
        B, M = retrieve.frozen_bundles(dom), retrieve.meta(dom)
        for base in bases:
            n = 0
            for qid in sorted(B):
                qs = retrieve.queries_for(B[qid], M[qid]["question_text"], "frozen")
                ref = [b["entry_id"] for b in _official_select(bank, ids, base, qs, None)]
                mine = [ids[i] for i in official_ids(bank, base, qs)]
                n += 1
                if ref != mine:
                    bad += 1
                    print(f"MISMATCH {dom} {base} {qid}: {ref} vs {mine}")
            print(f"[selftest] {dom} {base}: {n} questions checked")
    print(f"[selftest] mismatches: {bad}")
    if bad:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["selftest", "eval"])
    ap.add_argument("--bases", default="V,rrf:10:V+T:camel,rrf:10:V+K1,rrf:10:V+K0+K1")
    ap.add_argument("--evidence", default=str(step_keys.V2 / "evidence.json"))
    ap.add_argument("--variants", default="rrf:10:V+T:camel,rrf:10:V+T:camel~cons20")
    ap.add_argument("--out", default=str(step_keys.V2 / "reports/alloc.json"))
    a = ap.parse_args()
    if a.stage == "selftest":
        selftest(a.bases.split(","))
        return
    retrieve.select_question = select_question  # do_select looks the name up in the module at call time
    step_keys.do_eval(Path(a.evidence).expanduser(), a.variants.split(","), Path(a.out).expanduser())


if __name__ == "__main__":
    main()
