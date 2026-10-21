"""One interface over the AgentRunbook-R raw-state pool (one entry per state = radius-1 slice).

A retriever ranks ALL entries of a domain for one query:
    rank(query: Query, k) -> list[(entry_index, score)]   best first, len <= k
`entry_index` is the row of the official pool (raw_state_pool.jsonl / raw_state.npy), so every
method returns the same unit as the official dense search. A Query carries the query text, the
question it belongs to (for quoted anchors) and, for dense methods, a vector row id.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

E1 = Path("~/e1").expanduser()


@dataclass
class Query:
    text: str
    question_id: str = ""
    question_text: str = ""
    source: str = "raw"            # raw | question
    vec: np.ndarray | None = None  # official query embedding (normalised)


@dataclass
class EntryStore:
    """Fields of the official raw-state pool that the E2 retrievers need (streamed: the pool is
    ~230 MB for web, ~450 MB for enterprise)."""
    dom: str
    entry_ids: list[str] = field(default_factory=list)
    trajectory_ids: list[str] = field(default_factory=list)
    center: list[int] = field(default_factory=list)
    slice_states: list[list[int]] = field(default_factory=list)
    texts: list[str] | None = None      # slice_axtree_text (full, untruncated) if requested
    content: list[str] | None = None    # goal + action sequences + urls + slice text (for coverage)

    @classmethod
    def load(cls, dom: str, with_text: bool = False, with_content: bool = False) -> "EntryStore":
        st = cls(dom)
        st.texts = [] if with_text else None
        st.content = [] if with_content else None
        p = E1 / f"index/agentrunbook_r_{dom}_small/memory_state/pools/raw_state_pool.jsonl"
        with open(p) as fp:
            for line in fp:
                e = json.loads(line)
                st.entry_ids.append(e["entry_id"]); st.trajectory_ids.append(e["trajectory_id"])
                st.center.append(int(e["center_state_index"])); st.slice_states.append([int(x) for x in e["slice_state_indexes"]])
                if with_text:
                    st.texts.append(e["slice_axtree_text"])
                if with_content:
                    st.content.append("\n".join([e["goal"], e["full_action_sequence"], e["slice_action_sequence"],
                                                 "\n".join(e["slice_urls"]), e["slice_axtree_text"]]))
                del e
        st.index = {eid: i for i, eid in enumerate(st.entry_ids)}
        return st

    def __len__(self) -> int:
        return len(self.entry_ids)


class Retriever:
    name = "base"

    def scores(self, q: Query) -> np.ndarray:
        """Score of every entry (higher = better)."""
        raise NotImplementedError

    def rank(self, q: Query, k: int) -> list[tuple[int, float]]:
        """Top-k with the official ordering np.argsort(scores)[::-1] (as in _search_entries)."""
        s = self.scores(q)
        return [(i, float(s[i])) for i in official_topk(s, k)]


def official_topk(scores: np.ndarray, k: int) -> list[int]:
    """Exactly the official ordering: np.argsort(scores)[::-1][:top_k]."""
    return [int(i) for i in np.argsort(scores)[::-1][:k]]


class QueryVectors:
    """Official query embeddings written by e2_gpu.py queries (row per unique (source, text))."""

    def __init__(self, dom: str):
        d = E1 / "index_ext/queries" / dom
        self.V = np.load(d / "vectors.npy")
        self.row = {}
        with open(d / "queries.jsonl") as fp:
            for line in fp:
                r = json.loads(line)
                self.row.setdefault(r["text"], r["row"])  # same text -> same vector whatever the source

    def get(self, text: str) -> np.ndarray:
        return self.V[self.row[text]]
