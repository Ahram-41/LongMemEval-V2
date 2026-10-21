"""H: fusion of member retrievers over the whole pool, per query.
  rrf      : sum_m w_m / (k + rank_m(e)), ranks 1-based over all entries (k = 60 default)
  weighted : (1 - w) * minmax(member0) + w * minmax(member1); min-max over all entries of the
             query (a constant score list maps to 0). With members (Vfull, T), w is the lexical weight.
"""
from __future__ import annotations

import numpy as np

from e1.retrieval.base import Query, Retriever


def minmax(s: np.ndarray) -> np.ndarray:
    lo, hi = float(s.min()), float(s.max())
    return np.zeros_like(s) if hi - lo < 1e-12 else (s - lo) / (hi - lo)


def ranks(s: np.ndarray) -> np.ndarray:
    order = np.argsort(s)[::-1]
    r = np.empty(len(s), dtype=np.float64)
    r[order] = np.arange(1, len(s) + 1)
    return r


class Fuse(Retriever):
    def __init__(self, members: list[Retriever], mode: str = "rrf", k: float = 60, w: float = 0.5, weights=None):
        self.members, self.mode, self.k, self.w = members, mode, k, w
        self.weights = weights or [1.0] * len(members)
        self.name = (f"H-rrf{int(k)}" if mode == "rrf" else f"H-w{w:g}") + "(" + "+".join(m.name for m in members) + ")"

    def scores(self, q: Query) -> np.ndarray:
        S = [m.scores(q) for m in self.members]
        if self.mode == "rrf":
            return sum(wt / (self.k + ranks(s)) for wt, s in zip(self.weights, S))
        assert len(S) == 2
        return (1 - self.w) * minmax(S[0]) + self.w * minmax(S[1])
