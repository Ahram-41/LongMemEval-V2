#!/usr/bin/env python3
"""STALE scenario sets (CPU only; the dataset is one ~300 MB JSON array, so run in a CPU job, not on login).

  t1_69 : mem-test v2 selection: T1 scenarios, random.Random(0).shuffle, first 69 (same 69 as v1/v2)
  t1    : all 200 T1      t2 : all 200 T2      all : all 400 (dataset order)

Each row: uid, type, M_old, M_new, explanation, probing_queries, units (sid, date, role, text; flat over
sessions in order), old_sid/new_sid (by timestamp), j_old/j_new + match ratio (containment-first
localization, verbatim from stale_gen_v2.locate).
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.manifest import write_manifest  # noqa: E402
from e1.stale.common import DATA, PREP  # noqa: E402

norm = lambda s: " ".join(s.lower().replace("’", "'").split())  # noqa: E731


def locate(units, sid, text):
    """1) turn in session that literally contains the observation (first 50 normalized chars); 2) else best sentence-level fuzzy match over user turns."""
    key = norm(text)[:50]
    hit = [j for j, u in enumerate(units) if u["sid"] == sid and key in norm(u["text"])]
    if hit:
        return hit[0], 1.0
    cands = [j for j, u in enumerate(units) if u["sid"] == sid and u["role"] == "user"] or [j for j, u in enumerate(units) if u["sid"] == sid]

    def sent_ratio(j):
        sents = re.split(r"(?<=[.!?])\s+", units[j]["text"]) or [units[j]["text"]]
        return max(difflib.SequenceMatcher(None, s, text).ratio() for s in sents)
    j = max(cands, key=sent_ratio)
    return j, round(sent_ratio(j), 2)


def select(data, name):
    if name == "t1_69":
        t1 = [d for d in data if d["type"] == "T1"]
        random.Random(0).shuffle(t1)
        return t1[:69]
    if name in ("t1", "t2"):
        return [d for d in data if d["type"] == name.upper()]
    if name == "all":
        return data
    raise SystemExit(f"unknown set {name}")


def build(d):
    units = [{"sid": si, "date": d["timestamps"][si], "role": t["role"], "text": t["content"]}
             for si, sess in enumerate(d["haystack_session"]) for t in sess]
    a, b = d["relevant_session_index"]
    old_sid, new_sid = (a, b) if d["timestamps"][a] <= d["timestamps"][b] else (b, a)
    j_old, r_old = locate(units, old_sid, d["M_old"])
    j_new, r_new = locate(units, new_sid, d["M_new"])
    return {"uid": d["uid"], "type": d["type"], "M_old": d["M_old"], "M_new": d["M_new"], "explanation": d["explanation"],
            "probing_queries": d["probing_queries"], "old_sid": old_sid, "new_sid": new_sid,
            "j_old": int(j_old), "j_new": int(j_new), "match_old": r_old, "match_new": r_new, "units": units}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="t1_69,all")
    a = ap.parse_args()
    data = json.load(open(DATA))
    PREP.mkdir(parents=True, exist_ok=True)
    for name in a.sets.split(","):
        rows = [build(d) for d in select(data, name)]
        with open(PREP / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        exact = sum(r["match_old"] == 1.0 and r["match_new"] == 1.0 for r in rows)
        print(f"[stale prep] {name}: {len(rows)} scenarios, {sum(len(r['units']) for r in rows)} units, "
              f"both observations located exactly in {exact}", flush=True)
    write_manifest(PREP, inputs={"data": DATA}, params={"sets": a.sets}, name="manifest_prep.json")


if __name__ == "__main__":
    main()
