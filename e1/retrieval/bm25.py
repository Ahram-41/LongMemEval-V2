"""T: BM25 over each entry's full slice text (slice_axtree_text, untruncated).

Tokenizers (iteration knob):
  basic : lowercase, split on non-alphanumerics, keep numbers           (first version, per plan)
  camel : split camelCase / letter-digit boundaries first, then basic
  noid  : basic, but drop accessibility-tree node ids "[123]" before tokenizing
BM25 (Lucene form): idf = ln(1 + (N - df + 0.5) / (df + 0.5)); k1 = 1.2, b = 0.75; query terms
counted once each. The inverted index is cached under ~/e1/index_ext/bm25/<dom>/<tokenizer>.pkl.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter

import numpy as np

from e1.retrieval.base import E1, EntryStore, Query, Retriever

WORD = re.compile(r"[a-z0-9]+")
CAMEL1 = re.compile(r"([a-z])([A-Z])")
CAMEL2 = re.compile(r"([A-Za-z])([0-9])|([0-9])([A-Za-z])")
NODE_ID = re.compile(r"\[\d+\]")


def tokenize(text: str, mode: str = "basic") -> list[str]:
    if mode == "camel":
        text = CAMEL1.sub(r"\1 \2", text)
        text = CAMEL2.sub(lambda m: " ".join(g for g in m.groups() if g), text)
    elif mode == "noid":
        text = NODE_ID.sub(" ", text)
    return WORD.findall(text.lower())


def build_index(texts, tokenizer: str) -> dict:
    post: dict[str, tuple[list, list]] = {}
    dl = np.zeros(len(texts), dtype=np.float64)
    for i, t in enumerate(texts):
        c = Counter(tokenize(t, tokenizer))
        dl[i] = sum(c.values())
        for term, tf in c.items():
            p = post.get(term)
            if p is None:
                post[term] = p = ([], [])
            p[0].append(i); p[1].append(tf)
    return {"dl": dl, "post": {t: (np.array(a, np.int32), np.array(f, np.float32)) for t, (a, f) in post.items()}}


class BM25(Retriever):
    name = "T"

    def __init__(self, dom: str, store: EntryStore, tokenizer: str = "basic", k1: float = 1.2, b: float = 0.75):
        self.tok, self.k1, self.b = tokenizer, k1, b
        path = E1 / "index_ext/bm25" / dom / f"{tokenizer}.pkl"
        if path.exists():
            d = pickle.loads(path.read_bytes())
        else:
            assert store.texts is not None, "EntryStore.load(with_text=True) needed to build the index"
            d = build_index(store.texts, tokenizer)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL))
        self._set(d)

    def _set(self, d: dict) -> None:
        self.dl, self.post = d["dl"], d["post"]
        self.N = len(self.dl)
        mean = self.dl.mean() if self.N and self.dl.mean() > 0 else 1.0
        self.norm = self.k1 * (1 - self.b + self.b * self.dl / mean)

    @classmethod
    def from_texts(cls, texts: list[str], tokenizer: str = "camel", k1: float = 1.2, b: float = 0.75) -> "BM25":
        """In-memory index over a small corpus (E3: one LongMemEval_S haystack); same formula, no cache."""
        self = cls.__new__(cls)
        self.tok, self.k1, self.b = tokenizer, k1, b
        self._set(build_index(texts, tokenizer))
        return self

    def idf(self, term: str) -> float:
        p = self.post.get(term)
        df = 0 if p is None else len(p[0])
        return float(np.log(1 + (self.N - df + 0.5) / (df + 0.5)))

    def scores_terms(self, terms) -> np.ndarray:
        s = np.zeros(self.N)
        for t in dict.fromkeys(terms):
            p = self.post.get(t)
            if p is None:
                continue
            docs, tf = p
            s[docs] += self.idf(t) * tf * (self.k1 + 1) / (tf + self.norm[docs])
        return s

    def scores(self, q: Query) -> np.ndarray:
        return self.scores_terms(tokenize(q.text, self.tok))
