"""V-full: every state's full accessibility tree embedded in <=4,096-token pieces (no truncation).
Entry score = max over the pieces of the (up to 3) states in its slice. Same unit as V."""
from __future__ import annotations

import json

import numpy as np

from e1.retrieval.base import E1, EntryStore, Query, Retriever


class DenseFull(Retriever):
    name = "Vfull"

    def __init__(self, dom: str, store: EntryStore):
        d = E1 / "index_ext/dense_full" / dom
        self.P = np.load(d / "vectors.npy")
        keys, starts, n = [], [], 0
        with open(d / "pieces.jsonl") as fp:
            for line in fp:
                r = json.loads(line)
                k = (r["trajectory_id"], int(r["state_index"]))
                if not keys or keys[-1] != k:
                    keys.append(k); starts.append(n)
                n += 1
        assert len(set(keys)) == len(keys), "pieces of one state must be contiguous"
        assert self.P.shape[0] == n, (self.P.shape, n)
        self.starts = np.array(starts)
        sidx = {k: j for j, k in enumerate(keys)}
        m = max(len(s) for s in store.slice_states)
        self.ent = np.full((len(store), m), len(keys), dtype=np.int64)  # pad -> -inf column
        for e, (tid, sts) in enumerate(zip(store.trajectory_ids, store.slice_states)):
            for j, si in enumerate(sts):
                self.ent[e, j] = sidx[(tid, si)]

    def scores(self, q: Query) -> np.ndarray:
        ps = self.P @ q.vec
        st = np.maximum.reduceat(ps, self.starts)
        st = np.append(st, -np.inf)
        return st[self.ent].max(axis=1)


class DenseFullDedup(DenseFull):
    """Exploratory mechanism check (not a frozen arm): rank entries by V-full score, but an entry whose
    best-scoring piece is the same piece as that of a higher-ranked entry is skipped (falls through to
    the end, keeping its relative order). Implemented as a score transform so the official top-k and
    merge apply unchanged: duplicates get score - 10 (cosines are in [-1, 1])."""
    name = "Vfulldedup"

    def scores(self, q: Query) -> np.ndarray:
        ps = self.P @ q.vec
        # best piece per state
        st_max = np.maximum.reduceat(ps, self.starts)
        n_state = len(self.starts)
        best_piece = np.empty(n_state, dtype=np.int64)
        bounds = list(self.starts) + [len(ps)]
        for j in range(n_state):
            a, b = bounds[j], bounds[j + 1]
            best_piece[j] = a + int(np.argmax(ps[a:b]))
        st = np.append(st_max, -np.inf)
        bp = np.append(best_piece, -1)
        sc = st[self.ent]                       # [E, 3]
        arg = sc.argmax(axis=1)
        score = sc[np.arange(len(sc)), arg]
        piece = bp[self.ent[np.arange(len(sc)), arg]]
        out = score.copy()
        seen = set()
        for e in np.argsort(score)[::-1]:       # official ordering
            if piece[e] in seen:
                out[e] = score[e] - 10.0
            else:
                seen.add(piece[e])
        return out
