"""Can false-premise questions be detected from the question text alone (no retrieval)?
Bag of word uni+bigrams, L2 logistic regression (numpy, gradient descent), 5-fold stratified CV,
repeated with 5 seeds. Reports ROC AUC and the score at the checker's operating point."""
import json, re, random
import numpy as np
rows = [json.loads(l) for l in open(__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "questions.jsonl")]
y = np.array([r["question_type"].endswith("-abs") for r in rows], dtype=float)
def text(r):
    q = r["question"]
    return q if isinstance(q, str) else " ".join(x.get("value", "") for x in q if isinstance(x, dict) and x.get("type") == "text")
def toks(t):
    w = re.findall(r"[a-z]+", t.lower())
    return w + [a + "_" + b for a, b in zip(w, w[1:])]
docs = [toks(text(r)) for r in rows]
def auc(s, yy):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    pos = yy == 1; return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
def fit_predict(tr, te, lam=1.0, it=400, lr=0.5):
    vocab = {}
    for i in tr:
        for t in set(docs[i]): vocab[t] = vocab.get(t, 0) + 1
    vocab = {t: k for k, t in enumerate(t for t, c in vocab.items() if c >= 2)}
    def X(ids):
        M = np.zeros((len(ids), len(vocab)))
        for a, i in enumerate(ids):
            for t in docs[i]:
                if t in vocab: M[a, vocab[t]] = 1.0
        return M
    Xtr, Xte = X(tr), X(te); ytr = y[tr]
    w = np.zeros(Xtr.shape[1]); b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        g = Xtr.T @ (p - ytr) / len(tr) + lam * w / len(tr); gb = (p - ytr).mean()
        w -= lr * g; b -= lr * gb
    return Xte @ w + b, w, vocab
aucs = []; allw = {}
for seed in range(5):
    rng = random.Random(seed); pos = [i for i in range(len(y)) if y[i]]; neg = [i for i in range(len(y)) if not y[i]]
    rng.shuffle(pos); rng.shuffle(neg); folds = [pos[k::5] + neg[k::5] for k in range(5)]
    s = np.zeros(len(y))
    for k in range(5):
        te = folds[k]; tr = [i for j in range(5) if j != k for i in folds[j]]
        p, w, vocab = fit_predict(tr, te); s[te] = p
        if seed == 0:
            for t, idx in vocab.items(): allw[t] = allw.get(t, 0) + w[idx] / 5
    aucs.append(auc(s, y))
print(f"question-only classifier: ROC AUC {np.mean(aucs):.3f} (5 seeds: {', '.join(f'{a:.3f}' for a in aucs)})")
print("chance = 0.500; premise checker (sees retrieval) operating point: flags 0.258 of false-premise vs 0.173 of normal (CONTRADICTED)")
top = sorted(allw.items(), key=lambda x: -x[1])[:15]; bot = sorted(allw.items(), key=lambda x: x[1])[:10]
print("most false-premise-indicative n-grams:", ", ".join(f"{t}({v:.2f})" for t, v in top))
print("most normal-indicative n-grams:      ", ", ".join(f"{t}({v:.2f})" for t, v in bot))
for cue in ["between", "third", "besides", "second", "another", "which dropdown", "what is the"]:
    m = np.array([cue in text(r).lower() for r in rows])
    if m.sum(): print(f"  cue '{cue}': in {m.sum():3d} questions, false-premise rate {y[m].mean():.2f} (base rate {y.mean():.2f})")

# ---- format control: drop true/false and multiple-choice questions (never false-premise)
def is_tf_mc(r):
    t = text(r).lower()
    return bool(re.search(r"true or false|\(a\)|\ba\)|options?:|choose (one|from)|multiple choice|\ba\.\s", t)) or r["eval_function"].startswith("mc_")
keep = [i for i, r in enumerate(rows) if not is_tf_mc(r)]
yk = y[keep]
print(f"\nformat-controlled subset (no true/false, no multiple choice): n={len(keep)}, false-premise share {yk.mean():.2f}")
print("  eval functions kept:", sorted({rows[i]['eval_function'].split('|')[0] for i in keep}))
docs_all, y_all = docs, y
docs = [docs_all[i] for i in keep]; y = yk
aucs = []
for seed in range(5):
    rng = random.Random(seed); pos = [i for i in range(len(y)) if y[i]]; neg = [i for i in range(len(y)) if not y[i]]
    rng.shuffle(pos); rng.shuffle(neg); folds = [pos[k::5] + neg[k::5] for k in range(5)]
    s = np.zeros(len(y))
    for k in range(5):
        te = folds[k]; tr = [i for j in range(5) if j != k for i in folds[j]]
        p, _, _ = fit_predict(tr, te); s[te] = p
    aucs.append(auc(s, y))
print(f"  question-only ROC AUC within subset: {np.mean(aucs):.3f} (seeds {', '.join(f'{a:.3f}' for a in aucs)})")

# ---- checker AUC from the gate distributions (all 451; verdict order SUPPORTED < CONTRADICTED/NOT_FOUND)
def auc_from_dist(pos, neg):  # dicts score->prob, higher score = more 'false premise'
    a = 0.0
    for sp, pp in pos.items():
        for sn, pn in neg.items():
            a += pp * pn * (1.0 if sp > sn else 0.5 if sp == sn else 0.0)
    return a
fp = {"SUP": .188, "NF": .555, "CON": .258}; nm = {"SUP": .545, "NF": .266, "CON": .173, "ERR": .015}
for name, order in [("flag = NF or CON", {"SUP": 0, "ERR": 0, "NF": 1, "CON": 1}),
                    ("SUP < NF < CON", {"SUP": 0, "ERR": 0, "NF": 1, "CON": 2}),
                    ("flag = CON only", {"SUP": 0, "ERR": 0, "NF": 0, "CON": 1})]:
    P = {}; N = {}
    for k, v in fp.items(): P[order[k]] = P.get(order[k], 0) + v
    for k, v in nm.items(): N[order[k]] = N.get(order[k], 0) + v
    print(f"checker (sees retrieval) ROC AUC, {name}: {auc_from_dist(P, N):.3f}")
