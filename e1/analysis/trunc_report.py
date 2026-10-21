#!/usr/bin/env python3
"""How much of the raw-state section survives the official 200k truncation, per arm, and how long
are the retrieved pages? (interpreting R4: a coverage gain can be eaten by truncation)

For an arm with prompt rows (reader input after truncation) and a retrieved.jsonl (blocks before
truncation): per question, raw blocks retrieved vs raw blocks left in the reader context
(count of "### Raw state result" items) -> none lost / part lost / all lost.
For B1 the pre-truncation block list is known only for untruncated questions; truncated ones are
reported with the blocks left. Page length = token count of the entry's full slice text
(Qwen3-Embedding tokenizer, strata entrytext.pkl).
Usage: python -I e1/analysis/trunc_report.py
"""
import json
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402

E1 = Path("~/e1").expanduser()
W = E1 / "e2work"


def reader_blocks(prompt_rows: Path) -> dict:
    out = {}
    with open(prompt_rows) as fp:
        for line in fp:
            r = json.loads(line)
            n = sum(1 for x in r["memory_context"] if x.get("type") == "text" and x["value"].startswith("### Raw state result "))
            out[r["question_id"]] = (n, bool(r.get("memory_context_was_truncated")), r.get("memory_context_original_token_count"))
            del r
    return out


def main() -> None:
    for dom in ["web", "enterprise"]:
        ntok = dict(zip(*[pickle.loads((W / "strata" / f"{dom}_entrytext.pkl").read_bytes())[k] for k in ("entry_ids", "ntok")]))
        Q = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl")}
        print(f"\n=== {dom}")
        # page length of retrieved raw blocks (before truncation)
        for name, rows in [("B0/B1 (visible, complete only)", {q: [b["entry_id"] for b in r["baseline_raw_blocks"]] for q, r in Q.items()
                                                                if not r["context_truncated"]}),
                           ("V (frozen queries)", {r["question_id"]: [b["entry_id"] for b in r["raw_blocks"]] for r in iter_jsonl(W / f"retrieved/e2_v/{dom}.jsonl")}),
                           ("e2_single G:quoted:camel", {r["question_id"]: [b["entry_id"] for b in r["raw_blocks"]] for r in iter_jsonl(W / f"retrieved/e2_single/{dom}.jsonl")}),
                           ("e2_fusion rrf:10:V+T", {r["question_id"]: [b["entry_id"] for b in r["raw_blocks"]] for r in iter_jsonl(W / f"retrieved/e2_fusion/{dom}.jsonl")})]:
            toks = [ntok[e] for v in rows.values() for e in v]
            per_q = [sum(ntok[e] for e in v) for v in rows.values()]
            print(f"  {name:32s} questions {len(rows):3d} blocks {len(toks):4d}  page tokens median {statistics.median(toks):7.0f}"
                  f"  mean {statistics.mean(toks):7.0f}  | raw-section tokens per question median {statistics.median(per_q):7.0f}")
        # truncation of the reader context
        for arm, ret in [("b1_small", None), ("e2_fusion_small", "e2_fusion"), ("e2_single_small", "e2_single")]:
            p = E1 / f"runs/{arm}/{dom}_small/prompt_rows.jsonl"
            if not p.exists():
                continue
            RB = reader_blocks(p)
            pre = {}
            if ret:
                pre = {r["question_id"]: len(r["raw_blocks"]) for r in iter_jsonl(W / f"retrieved/{ret}/{dom}.jsonl")}
            else:
                pre = {q: len(r["baseline_raw_blocks"]) for q, r in Q.items() if not r["context_truncated"]}
            c = Counter()
            for q, (n, tr, orig) in RB.items():
                if not tr:
                    c["not truncated"] += 1
                elif q not in pre:
                    c["truncated, pre-truncation blocks unknown, 0 left" if n == 0 else "truncated, pre-truncation blocks unknown, some left"] += 1
                elif n == 0:
                    c["truncated, ALL raw blocks lost"] += 1
                elif n < pre[q]:
                    c["truncated, PART of raw blocks lost"] += 1
                else:
                    c["truncated, raw section intact"] += 1
            orig = [o for (_, _, o) in RB.values() if o]
            print(f"  {arm:16s} n={len(RB)}  " + "; ".join(f"{k}: {v}" for k, v in sorted(c.items()))
                  + f"  | pre-truncation tokens median {statistics.median(orig):.0f}")


if __name__ == "__main__":
    main()
