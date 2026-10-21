#!/usr/bin/env python3
"""E2 reader comparison: arms vs B1 (exact McNemar), on a chosen question set, broken down by
normal / false-premise and by question_type, plus how many questions were rerun.
Sets: complete (queries recovered from an untruncated B0 context, 344), regenerated (107), all.
Usage: python3 e1/analysis/e2_compare.py ~/e1/runs b1_small e2_x_merged_small [...] --set complete
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl, load_rows  # noqa: E402
from e1.common.stats import mcnemar_exact, paired_counts  # noqa: E402

E1 = Path("~/e1").expanduser()
TYPES = ["static-environment", "dynamic-environment", "procedure", "errors-gotchas",
         "static-environment-abs", "dynamic-environment-abs", "procedure-abs"]


def qset(which: str) -> set[str]:
    out = set()
    for dom in ["web", "enterprise"]:
        for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl"):
            regen = r["context_truncated"] or r["missing_detected"]
            if which == "all" or (which == "complete") != regen:
                out.add(r["question_id"])
    return out


def rerun_ids(root: Path, arm: str) -> set[str]:
    base = arm.replace("_merged", "")
    s = set()
    for d in ["web_small", "enterprise_small"]:
        p = root / base / d / "rerun_ids.txt"
        if p.exists():
            s |= {x for x in p.read_text().split(",") if x}
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--set", default="complete", choices=["complete", "regenerated", "all"])
    ap.add_argument("--by-domain", action="store_true")
    ap.add_argument("--cov", action="append", default=[],
                    help="<arm>=<retrieved dir>: layer 4, answer-coverage change vs B1 blocks x correctness")
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    keep = ("question_id", "score_bool", "is_abstention_problem", "question_type")
    D = {x: load_rows(root, x, keep) for x in a.arms}
    ref = a.arms[0]
    S0 = sorted(qset(a.set) & set(D[ref]))
    doms = {}
    for dom in ["web", "enterprise"]:
        for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl"):
            doms[r["question_id"]] = dom
    splits = [("both", S0)] + ([("web", [q for q in S0 if doms[q] == "web"]), ("enterprise", [q for q in S0 if doms[q] == "enterprise"])]
                               if a.by_domain else [])
    for dname, S in splits:
        print(f"\n=== set={a.set} domain={dname} n={len(S)}; reference {ref}")
        groups = [("all", S), ("normal", [q for q in S if not D[ref][q]["is_abstention_problem"]]),
                  ("false-premise", [q for q in S if D[ref][q]["is_abstention_problem"]])]
        groups += [(t, [q for q in S if D[ref][q]["question_type"] == t]) for t in TYPES]
        for x in a.arms[1:]:
            rr = rerun_ids(root, x)
            print(f"-- {x}: rerun {len(set(S) & rr)}/{len(S)} questions (others reuse B1 outputs)")
            for g, ids in groups:
                ids = [q for q in ids if q in D[x]]
                if not ids:
                    continue
                k0 = sum(bool(D[ref][q]["score_bool"]) for q in ids); k1 = sum(bool(D[x][q]["score_bool"]) for q in ids)
                b, c = paired_counts(D[ref], D[x], ids)
                print(f"   {g:24s} n={len(ids):3d}  {ref} {k0:3d} ({k0/len(ids):.3f})  {x} {k1:3d} ({k1/len(ids):.3f})  "
                      f"d={(k1-k0)/len(ids):+.3f}  fixed {c} broke {b}  McNemar p={mcnemar_exact(b, c):.3f}")
    for spec in a.cov:
        arm, ret = spec.split("=", 1)
        layer4(root, ref, arm, Path(ret).expanduser(), set(S0), D)


def layer4(root: Path, ref: str, arm: str, ret_dir: Path, S: set[str], D: dict) -> None:
    """§十一 layer 4: coverage change (arm blocks vs B1's blocks, answer-phrase proxy) x arm correctness."""
    from e1.analysis.retrieval_eval import Content, label_question
    from e1.pipeline.retrieve import meta
    cells = {"gain": [0, 0, 0], "loss": [0, 0, 0], "same-covered": [0, 0, 0], "same-uncovered": [0, 0, 0]}
    for dom in ["web", "enterprise"]:
        C, M = Content(dom), meta(dom)
        Q = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl")}
        R = {r["question_id"]: r for r in iter_jsonl(ret_dir / f"{dom}.jsonl")}
        ids = [q for q in R if q in S and q in D[arm]]
        C.load({b["entry_id"] for q in ids for b in R[q]["raw_blocks"] + Q[q]["baseline_raw_blocks"]})
        for q in ids:
            lx = label_question(M[q], [b["entry_id"] for b in R[q]["raw_blocks"]], C)
            lb = label_question(M[q], [b["entry_id"] for b in Q[q]["baseline_raw_blocks"]], C)
            if "ans" not in lx:
                continue
            k = "gain" if lx["ans"] and not lb["ans"] else "loss" if lb["ans"] and not lx["ans"] else \
                "same-covered" if lx["ans"] else "same-uncovered"
            cells[k][0] += bool(D[arm][q]["score_bool"]); cells[k][1] += 1; cells[k][2] += bool(D[ref][q]["score_bool"])
    print(f"-- layer 4 {arm}: answer coverage vs B1 blocks x {arm} correct (phrase-answer normal questions)")
    for k, (c, n, c0) in cells.items():
        print(f"   coverage {k:15s} n={n:3d}  {arm} correct {c}  ({ref} correct {c0})")


if __name__ == "__main__":
    main()
