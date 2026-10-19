"""Validate the minimal-pair design: out-of-fold question-only scores should NOT separate pair members."""
import json, re, random, math
import numpy as np
exec(open("qonly.py").read().split("aucs = []; allw = {}")[0])   # reuse rows, y, docs, text, fit_predict, auc
pairs = json.load(open("pairs.json")); idx = {r["id"]: i for i, r in enumerate(rows)}
def sign_p(b, c):
    b, c = int(b), int(c); n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
for th in [0.6, 0.8]:
    P = [(idx[p["abs_id"]], idx[p["normal_id"]]) for p in pairs if p["jaccard"] >= th]
    wins = []
    for seed in range(5):
        # out-of-fold scores, keeping BOTH members of a pair in the same fold (no leakage across the pair)
        rng = random.Random(seed); groups = {}
        for a, n in P: groups[a] = groups[n] = len(groups)
        units = {}
        for i in range(len(y)):
            g = groups.get(i, ("s", i)); units.setdefault(g, []).append(i)
        keys = list(units); rng.shuffle(keys); folds = [[i for k in keys[f::5] for i in units[k]] for f in range(5)]
        s = np.zeros(len(y))
        for f in range(5):
            te = folds[f]; tr = [i for j in range(5) if j != f for i in folds[j]]
            pr, _, _ = fit_predict(tr, te); s[te] = pr
        b = sum(s[a] > s[n] for a, n in P); c = sum(s[a] < s[n] for a, n in P)
        wins.append((b, c))
    bm = np.mean([w[0] for w in wins]); cm = np.mean([w[1] for w in wins])
    print(f"Jaccard >= {th}: {len(P)} pairs. question-only score higher for the FALSE member in {bm:.1f}, for the NORMAL member in {cm:.1f} (mean of 5 seeds); "
          f"pairwise accuracy {bm/(bm+cm):.2f}; sign-test p (seed 0) = {sign_p(*wins[0]):.3f}")
