#!/usr/bin/env python3
"""Minimal-pair test: does the premise checker / reader use evidence, or only question phrasing?

Each false-premise question is paired with its most similar normal question (e1/analysis/minimal_pairs.json,
Jaccard over question words, format instructions stripped). Within a pair the phrasing prior is
(nearly) constant, so any systematic difference must come from the retrieved evidence.

For pairs with Jaccard >= threshold it reports:
  checker: flagged(false member) vs flagged(normal member) -> discordant counts + exact sign test
  each arm: correct(false member) and correct(normal member)
Usage: python e1/analysis/pair_test.py ~/e1/runs [threshold=0.6]
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import DOMAINS, iter_jsonl  # noqa: E402
from e1.common.stats import sign_test as sign_p  # noqa: E402

root = Path(sys.argv[1]).expanduser(); th = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8  # 0.8: wording prior n.s. within pairs (pair_validate.py)
pairs = [p for p in json.load(open(Path(__file__).with_name("minimal_pairs.json"))) if p["jaccard"] >= th]
def load(arm, fname="per_question.jsonl"):
    out = {}
    for d in DOMAINS:
        p = root / arm / d / fname
        if p.exists():
            for r in iter_jsonl(p):
                out[r["question_id"]] = r
    return out
ver = load("p_small", "premise_check.jsonl")
print(f"pairs with Jaccard >= {th}: {len(pairs)}")
for name, flag in [("CON", lambda v: v["overall"] == "CONTRADICTED"),
                   ("NF or CON", lambda v: v["overall"] in ("CONTRADICTED", "NOT_FOUND"))]:
    ps = [p for p in pairs if p["abs_id"] in ver and p["normal_id"] in ver]
    b = sum(flag(ver[p["abs_id"]]) and not flag(ver[p["normal_id"]]) for p in ps)
    c = sum(flag(ver[p["normal_id"]]) and not flag(ver[p["abs_id"]]) for p in ps)
    both = sum(flag(ver[p["abs_id"]]) and flag(ver[p["normal_id"]]) for p in ps)
    print(f"checker {name:9s}: only false member flagged {b}, only normal member flagged {c}, both {both}, n={len(ps)}, sign-test p={sign_p(b, c):.3f}")
for arm in ["agentrunbook_r_small", "b0r_small", "b1_small", "pa_merged_small", "ps_merged_small", "pnf_merged_small"]:
    R = load(arm)
    ps = [p for p in pairs if p["abs_id"] in R and p["normal_id"] in R]
    if not ps: continue
    fa = sum(R[p["abs_id"]]["score_bool"] for p in ps); no = sum(R[p["normal_id"]]["score_bool"] for p in ps)
    print(f"{arm:22s} n={len(ps)}  false member correct {fa:3d} ({fa/len(ps):.2f})  normal member correct {no:3d} ({no/len(ps):.2f})")
