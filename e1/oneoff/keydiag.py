"""Diagnostic (read-only): for the 13 true misses, show the queries, and under K1 (step keys alone) the best query's
top-3 keys vs the best-ranked evidence entry's key. Answers: does the key text carry what the query asks for?"""
import json, os, sys
from pathlib import Path
V2 = Path.home() / "e1/v2work"
os.environ.setdefault("E2WORK", str(V2))
sys.path.insert(0, str(Path.home() / "e1/LongMemEval-V2"))
sys.path.insert(0, str(Path.home() / "e1/LongMemEval-V2/e1/oneoff"))
import numpy as np
from step_keys import evidence_sets, DOMS
from e1.pipeline.retrieve import ScoreBank, entry_ids, frozen_bundles, meta, queries_for

E = "e9fde9fe deb008ed 901b7d17 01307e07 f4b2b1fb 15e5efa6 289aea4a adddeaa5 19c641bd 5cb7289b 25b00876 e77d1287 a45edf3e".split()
rules = json.loads((V2 / "evidence.json").read_text())["questions"]
ev = evidence_sets(rules)
for dom in DOMS:
    keys = {json.loads(l)["entry_id"]: json.loads(l)["key"] for l in open(V2 / f"keys/{dom}/llm.jsonl")}
    bank, ids = ScoreBank(dom), entry_ids(dom)
    pos = {e: i for i, e in enumerate(ids)}
    B, M = frozen_bundles(dom), meta(dom)
    for q8 in E:
        if q8 not in ev or rules[q8]["domain"] != dom:
            continue
        qid = next(q for q in B if q.startswith(q8))
        qs = queries_for(B[qid], M[qid]["question_text"], "frozen")
        print("=" * 100); print(q8, dom, "|", M[qid]["question_text"][:400])
        for g, S in ev[q8].items():
            cols = [pos[e] for e in S if e in pos]
            best = None
            for t in qs:
                for v in ("K1", "V"):
                    s = bank.scores(v, t)
                    order = np.argsort(-s, kind="stable")
                    rank = np.empty(len(s), int); rank[order] = np.arange(1, len(s) + 1)
                    c = min(cols, key=lambda c: rank[c])
                    if v == "K1" and (best is None or rank[c] < best[0]):
                        best = (rank[c], t, order[:3], c, len(S))
            r, t, top, c, nS = best
            print(f"-- group {g}: {nS} evidence entries in pool; best K1 rank {r} via query: {t!r}")
            for i, j in enumerate(top, 1):
                print(f"   top{i}{' [EVIDENCE]' if j in cols else ''} {ids[j]}: {keys.get(ids[j],'')[:330]!r}")
            print(f"   EVIDENCE {ids[c]}: {keys.get(ids[c],'')[:450]!r}")
        print("   all queries:", [q if isinstance(q, str) else q for q in qs][:6])
