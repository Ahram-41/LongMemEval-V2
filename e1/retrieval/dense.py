"""V: the official dense search. Stored (truncated) raw-state embeddings x official query vector."""
from __future__ import annotations

import numpy as np

from e1.retrieval.base import E1, Query, Retriever


class Dense(Retriever):
    name = "V"

    def __init__(self, dom: str):
        self.E = np.load(E1 / f"index/agentrunbook_r_{dom}_small/memory_state/embeddings/raw_state.npy")

    def scores(self, q: Query) -> np.ndarray:
        return self.E @ q.vec
