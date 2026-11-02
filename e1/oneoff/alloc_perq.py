"""Post hoc: per-query ranks of the best evidence entry (fusion), to test the 'evidence is shared across queries' premise of A."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("E2WORK", str(Path.home() / "e1/v2work"))
sys.path.insert(0, str(Path.home() / "e1/LongMemEval-V2")); sys.path.insert(0, str(Path.home() / "e1/LongMemEval-V2/e1/oneoff"))
import numpy as np
from step_keys import evidence_sets, DOMS
from e1.pipeline.retrieve import ScoreBank, entry_ids, frozen_bundles, meta, queries_for
from e1.retrieval.fuse import ranks
rules = json.loads((Path.home() / "e1/v2work/evidence.json").read_text())["questions"]
ev = evidence_sets(rules); base = sys.argv[1]
for dom in DOMS:
    bank, ids = ScoreBank(dom), entry_ids(dom); pos = {e: i for i, e in enumerate(ids)}
    B, M = frozen_bundles(dom), meta(dom)
    for q8, G in ev.items():
        if rules[q8]["domain"] != dom: continue
        qid = next(q for q in B if q.startswith(q8)); qs = queries_for(B[qid], M[qid]["question_text"], "frozen")
        R = [ranks(np.asarray(bank.scores(base, t), np.float64)) for t in qs]
        print(q8, {g: [int(min(r[pos[e]] for e in S)) for r in R] for g, S in G.items()})
