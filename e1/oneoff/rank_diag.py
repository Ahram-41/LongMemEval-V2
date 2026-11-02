"""Diagnostic (not pre-registered as a decision): best rank of each evidence group under each variant, min over the question's frozen queries."""
import json, os, sys
from pathlib import Path
V2 = Path.home() / "e1/v2work"
os.environ.setdefault("E2WORK", str(V2))
sys.path.insert(0, str(Path.home() / "e1/LongMemEval-V2"))
sys.path.insert(0, str(Path.home() / "e1/LongMemEval-V2/e1/oneoff"))
import numpy as np
from step_keys import evidence_sets, DOMS
from e1.pipeline.retrieve import ScoreBank, entry_ids, frozen_bundles, meta, queries_for

rules = json.loads((V2 / "evidence.json").read_text())["questions"]
ev = evidence_sets(rules)
variants = sys.argv[1].split(",")
out = {}
for dom in DOMS:
    bank, ids = ScoreBank(dom), entry_ids(dom)
    pos = {e: i for i, e in enumerate(ids)}
    B, M = frozen_bundles(dom), meta(dom)
    for q8, G in ev.items():
        if rules[q8]["domain"] != dom:
            continue
        qid = next(q for q in B if q.startswith(q8))
        qs = queries_for(B[qid], M[qid]["question_text"], "frozen")
        row = {"n_queries": len(qs)}
        for v in variants:
            best = {}
            for g, S in G.items():
                cols = [pos[e] for e in S if e in pos]
                r_best = None
                for t in qs:
                    s = bank.scores(v, t)
                    order = np.argsort(-s, kind="stable")
                    rank = np.empty(len(s), int); rank[order] = np.arange(1, len(s) + 1)
                    r = int(rank[cols].min()) if cols else None
                    if r is not None and (r_best is None or r < r_best):
                        r_best = r
                best[g] = r_best
            row[v] = best
        out[q8] = row
print(json.dumps(out, indent=1))
