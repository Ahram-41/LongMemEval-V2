"""G: grep-style exact matching. Terms = quoted phrases (from the question and the query) and/or
query keywords (query tokens minus stopwords, length >= 2). Entry score = number of distinct terms
it contains (case-insensitive exact match: keywords as whole tokens, phrases as substrings of the
full slice text), ties broken by BM25 (scaled below 1 so it never outweighs one more matched term).
Term source (iteration knob): quoted | keywords | both.
"""
from __future__ import annotations

import numpy as np

from e1.common.anchors import anchors
from e1.retrieval.base import EntryStore, Query, Retriever
from e1.retrieval.bm25 import BM25, tokenize

STOP = set("""a an the of to in on at for from by with and or not no is are was were be been being this that these those
it its as into than then there here what which who whom whose when where why how do does did can could should would
will shall may might must i you he she we they me my your our their them his her us page pages state states click
clicked clicking show shows shown showing user users after before about any all some each other same also only just
via list button text field option options""".split())


class Grep(Retriever):
    name = "G"

    def __init__(self, store: EntryStore, bm25: BM25, source: str = "both"):
        assert store.texts is not None
        self.st, self.bm, self.source = store, bm25, source
        self.low: dict[int, str] = {}

    def _low(self, i: int) -> str:
        t = self.low.get(i)
        if t is None:
            t = self.low[i] = self.st.texts[i].lower()
        return t

    def terms(self, q: Query) -> tuple[list[str], list[str]]:
        phrases = anchors(q.question_text) + [a for a in anchors(q.text) if a not in anchors(q.question_text)]
        kws = [w for w in dict.fromkeys(tokenize(q.text, self.bm.tok)) if len(w) >= 2 and w not in STOP]
        if self.source == "quoted":
            return phrases, []
        if self.source == "keywords":
            return [], kws
        return phrases, kws

    def scores(self, q: Query) -> np.ndarray:
        phrases, kws = self.terms(q)
        hits = np.zeros(self.bm.N)
        for w in kws:
            p = self.bm.post.get(w)
            if p is not None:
                hits[p[0]] += 1
        for ph in phrases:
            toks = tokenize(ph, self.bm.tok)
            cand = None
            for t in toks:  # candidate entries contain every token of the phrase
                p = self.bm.post.get(t)
                s = set() if p is None else set(p[0].tolist())
                cand = s if cand is None else cand & s
            for i in (cand if cand is not None else range(self.bm.N)):
                if ph in self._low(i):
                    hits[i] += 1
        tb = self.bm.scores_terms(tokenize(q.text, self.bm.tok) + [t for ph in phrases for t in tokenize(ph, self.bm.tok)])
        return hits + 0.999 * tb / (tb.max() + 1e-9)
