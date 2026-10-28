#!/usr/bin/env python3
"""Port check: cluster contexts (RUNS/<arm>/gen.jsonl) vs mem-test v2 local contexts (stale_gen_v2.jsonl, copied to
WORK/ref/). Same scenarios, localization and retrieval should give the same unit indices per condition; differences
can only come from bge numerics (MPS vs CUDA) at near-ties. Also reports v2 C0/O_new_only labels for orientation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.stale.common import DIMS, RUNS, WORK, iter_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref", default=str(WORK / "ref/stale_gen_v2.jsonl"))
    a = ap.parse_args()
    ref = {r["uid"]: r for r in iter_jsonl(a.ref)}
    new = {r["uid"]: r for r in iter_jsonl(RUNS / a.arm / "gen.jsonl")}
    common = [u for u in new if u in ref]
    print(f"scenarios: cluster {len(new)}, local v2 {len(ref)}, common {len(common)}")
    loc = sum(ref[u]["j_old"] == new[u]["j_old"] and ref[u]["j_new"] == new[u]["j_new"] for u in common)
    print(f"localization (j_old, j_new) identical: {loc}/{len(common)}")
    conds = list(new[common[0]]["probes"]["dim1_query"]["ctx"])
    for c in conds:
        same_list = same_set = overlap = n = 0
        for u in common:
            for d in DIMS:
                x, y = new[u]["probes"][d]["ctx"][c], ref[u]["probes"][d]["ctx"][c]
                n += 1
                same_list += x == y
                same_set += set(x) == set(y)
                overlap += len(set(x) & set(y))
        print(f"  {c:>13}: identical list {same_list}/{n}, identical set {same_set}/{n}, mean overlap {overlap / n:.2f}/10")
    meta = sum(all(new[u]["probes"][d][k] == ref[u]["probes"][d][k] for k in ("old_in_top", "new_in_top", "n_old_sess", "n_new_sess"))
               for u in common for d in DIMS)
    print(f"probe meta (old/new in top, session counts) identical: {meta}/{len(common) * 3}")


if __name__ == "__main__":
    main()
