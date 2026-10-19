#!/usr/bin/env python3
"""Paired comparison of E1 arms on LME-V2 Small (same questions, same retrieval).

Usage: python e1/compare_arms.py ~/e1/runs  agentrunbook_r_small b0r_small b1_small p_small
The first arm is the reference. For every other arm it prints, per subset
(all / normal / false-premise / per question_type):
  accuracy, delta vs reference, exact two-sided McNemar p on discordant pairs,
  cap-hit rate (completion == 20000 tokens), UNKNOWN rate,
plus mean query latency. Arms with a missing per_question.jsonl are skipped.
"""
import json
import math
import sys
from pathlib import Path

CAP = 20000
KEEP = ("question_id", "score_bool", "is_unknown", "is_abstention_problem", "question_type", "usage", "memory_query_duration_seconds")


def load(run_root: Path, arm: str):
    rows = {}
    for d in ["web_small", "enterprise_small"]:
        p = run_root / arm / d / "per_question.jsonl"
        if not p.exists():
            return None
        for line in open(p):
            r = json.loads(line)
            rows[r["question_id"]] = {k: r.get(k) for k in KEEP}
            del r
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def stats(rows, ids):
    n = len(ids)
    acc = sum(rows[q]["score_bool"] for q in ids) / n
    cap = sum((rows[q].get("usage") or {}).get("completion_tokens", 0) >= CAP for q in ids) / n
    unk = sum(rows[q]["is_unknown"] for q in ids) / n
    return acc, cap, unk


def main():
    root = Path(sys.argv[1]).expanduser()
    arms = sys.argv[2:]
    data = {a: load(root, a) for a in arms}
    ref_name = arms[0]
    ref = data[ref_name]
    assert ref is not None, f"reference arm {ref_name} missing"
    qids = sorted(ref)
    subsets = {"all": qids,
               "normal": [q for q in qids if not ref[q]["is_abstention_problem"]],
               "false-premise": [q for q in qids if ref[q]["is_abstention_problem"]]}
    for qt in sorted({ref[q]["question_type"] for q in qids}):
        subsets[qt] = [q for q in qids if ref[q]["question_type"] == qt]

    print(f"reference: {ref_name}")
    header = f"{'subset':26s} {'n':>4s} " + " ".join(f"{a[:14]:>34s}" for a in arms)
    print(header)
    print(f"{'':26s} {'':>4s} " + " ".join(f"{'acc  (d, p_mcn) cap unk':>34s}" for _ in arms))
    for name, ids in subsets.items():
        cells = []
        for a in arms:
            rows = data[a]
            if rows is None or any(q not in rows for q in ids):
                cells.append(f"{'n/a':>34s}")
                continue
            acc, cap, unk = stats(rows, ids)
            if a == ref_name:
                cells.append(f"{acc:.3f} {'(ref)':>14s} {cap:.2f} {unk:.2f}".rjust(34))
            else:
                b = sum(ref[q]["score_bool"] and not rows[q]["score_bool"] for q in ids)
                c = sum((not ref[q]["score_bool"]) and rows[q]["score_bool"] for q in ids)
                d = acc - stats(ref, ids)[0]
                cells.append(f"{acc:.3f} ({d:+.3f}, {mcnemar_exact(b, c):.3f}) {cap:.2f} {unk:.2f}".rjust(34))
        print(f"{name:26s} {len(ids):4d} " + " ".join(cells))
    print("\nmean memory query latency (s), includes the premise check for P:")
    for a in arms:
        rows = data[a]
        if rows is None:
            print(f"  {a}: n/a")
            continue
        lat = [rows[q].get("memory_query_duration_seconds") or 0.0 for q in qids if q in rows]
        print(f"  {a}: {sum(lat)/len(lat):.1f}")


if __name__ == "__main__":
    main()
