#!/usr/bin/env python3
"""E2 §十一 layer 3: sample 5 gains and 5 losses (answer-phrase proxy, vs V) per method on web,
seed 0, and print what is needed to label each with ONE fixed reason (labels fixed in advance):
  tail evidence recovered | exact term hit | paraphrase miss | boilerplate flood | tie broken |
  query lost (truncation) | other
Printed per question: type, query source, question, answer phrases, the queries, and for the
covering block(s) of the winning side: entry id, the query and rank that brought it in, whether the
answer phrases lie inside that entry's embedded prefix (official 40k-char cut + 4,096 tokens), and
the losing side's score of that entry. Labels are written by hand into the night log.
Usage: python -I e1/analysis/reason_sample.py --variants Vfull,T:basic --domain web --set complete
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.analysis.retrieval_eval import Content, answer_phrases, eval_variant, question_set  # noqa: E402
from e1.analysis.strata import CHAR_CAP, TOK_CAP, TOKENIZER, truncate_middle  # noqa: E402
from e1.pipeline.retrieve import ScoreBank, do_select, meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="Vfull,T:basic")
    ap.add_argument("--domain", default="web")
    ap.add_argument("--set", default="complete")
    ap.add_argument("--ref", default="V")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    from e1.analysis.retrieval_eval import _official_metrics
    norm = _official_metrics()[0]
    tok = Tokenizer.from_file(str(TOKENIZER))
    dom = a.domain
    C, M, bank = Content(dom), meta(dom), ScoreBank(dom)
    qids = question_set(dom, a.set)
    Lr = eval_variant(dom, a.ref, "frozen", qids, C, bank, C.ids)
    Rr = {r["question_id"]: r for r in do_select(dom, a.ref, "frozen", None, bank=bank, ids=C.ids, sim_variant=None)}
    texts = {}

    def slice_text(eid):
        if eid not in texts:
            with open(Path("~/e1").expanduser() / f"index/agentrunbook_r_{dom}_small/memory_state/pools/raw_state_pool.jsonl") as fp:
                for line in fp:
                    if f'"entry_id": "{eid}"' in line[:80]:
                        texts[eid] = json.loads(line)["slice_axtree_text"]; break
        return texts[eid]

    def in_prefix(eid, ph):
        t = truncate_middle(slice_text(eid), CHAR_CAP)
        enc = tok.encode(t, add_special_tokens=False)
        emb = t if len(enc.ids) <= TOK_CAP else t[: enc.offsets[TOK_CAP - 1][1]]
        e = " " + norm(emb) + " "
        return all(f" {p} " in e for p in ph)

    for v in a.variants.split(","):
        Lx = eval_variant(dom, v, "frozen", qids, C, bank, C.ids)
        Rx = {r["question_id"]: r for r in do_select(dom, v, "frozen", None, bank=bank, ids=C.ids, sim_variant=None)}
        gains = sorted(q for q in qids if "ans" in Lx[q] and Lx[q]["ans"] and not Lr[q]["ans"])
        losses = sorted(q for q in qids if "ans" in Lx[q] and Lr[q]["ans"] and not Lx[q]["ans"])
        rng = random.Random(0)
        for kind, pool, win, lose in [("GAIN", gains, Rx, Rr), ("LOSS", losses, Rr, Rx)]:
            pick = rng.sample(pool, min(a.k, len(pool)))
            print(f"\n######## {v} vs {a.ref}: {kind} {len(pick)} of {len(pool)}")
            for q in pick:
                m = M[q]; ph = answer_phrases(m)
                print(f"\n[{kind}] {q} {m['question_type']} source={win[q]['query_source']}\n  Q: {m['question_text'][:300]}\n  answer phrases: {ph}")
                for i, qq in enumerate(win[q]["queries"]):
                    print(f"  query{i}: {qq[:160]}")
                lose_ids = {b['entry_id'] for b in lose[q]["raw_blocks"]}
                for b in win[q]["raw_blocks"]:
                    if b["entry_id"] in lose_ids:
                        continue
                    has = all(f" {p} " in C.norm[b["entry_id"]] for p in ph)
                    if not has:
                        continue
                    other = a.ref if kind == "GAIN" else v
                    o = bank.scores(other, b["query"])
                    rank_o = int((o > o[C.idx[b["entry_id"]]]).sum()) + 1
                    ties = int((o == o[C.idx[b["entry_id"]]]).sum()) - 1
                    print(f"  covering block {b['entry_id']} via query '{b['query'][:80]}' rank {b['rank']}; "
                          f"answer in embedded prefix: {in_prefix(b['entry_id'], ph)}; rank under {other}: {rank_o} (exact ties {ties})")
                print(f"  {a.ref} blocks: {[b['entry_id'] for b in Rr[q]['raw_blocks']]}")
                print(f"  {v} blocks: {[b['entry_id'] for b in Rx[q]['raw_blocks']]}")


if __name__ == "__main__":
    main()
