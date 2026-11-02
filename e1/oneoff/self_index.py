#!/usr/bin/env python3
"""Self-Index-style index-key evolution on the AgentRunbook-R raw-state pool (retrieval only, no reader).

Pre-registration: CP5101/experiments/self-index-plan-2026-11-02.md (written before any output).
Paper: SELF-INDEX 2609.19656, Algorithms 1-3 and the prompts of Tables 6-10 (used verbatim below).
Deviations from the paper are marked DEV in the plan and in comments.

Index: K(d) = {original key (the official V row of raw_state.npy, never revised)} U generated keys (<= MAX_GEN).
Document score for a query = max over its keys of the cosine between the query vector and the key vector.

Work dir $V2WORK/si/<dom>/:
  keys.json          {entry_id: [generated key texts]}            (current index)
  keyvec.npz         doc-form and query-form vectors of every key text ever embedded (cache)
  queries.jsonl      accepted simulated queries (append-only; = Q_used)
  iter_<t>.json      per-iteration log (counts, pass rates, a few examples)
  SI.npy             -> $V2WORK/scores/<dom>/SI.npy  max(V, keys)   for step_keys.py eval
  SIk.npy            -> $V2WORK/scores/<dom>/SIk.npy keys only (diagnostic)

Stages:  iterate --iters N    GPU (reader server = optimizer/judges, embedding server)
         score                CPU + embedding server
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
E1 = Path(os.environ.get("E1", "~/e1")).expanduser()
V2 = Path(os.environ.get("V2WORK", str(E1 / "v2work")))
os.environ.setdefault("E2WORK", str(V2))
TAG = os.environ.get("SITAG", "si")        # work dir v2work/<TAG>/<dom>; matrices <MAT>.npy / <MAT>k.npy
MAT = os.environ.get("SIMAT", "SI")
SIM = os.environ.get("SISIM", "task")      # task = paper Table 9; ui = interface-fact variant (self-index-ui-plan-2026-11-03.md)

TOP_K = 30            # retrieval depth during optimisation and for the Specificity check
MAX_GEN = 9           # generated keys per document (paper: m_max = 10 including the original)
BATCH_Q = 128         # optimisation queries per iteration
SIM_DOCS = 80         # documents sampled per simulator call (DEV: paper does not fix n in B.3)
SIM_M = 3             # queries requested per sampled document
JACCARD_T = 0.8
JUDGE_MIN = 2
DOC_AXTREE_CHARS = 12000   # DEV: the LLM sees goal+actions+urls+K0 struct+AXTree head instead of the whole page
STRUCT_CHARS = 2500
NEIGH_CHARS = 300          # DEV: another document's original-text key is shown as its K0 struct head
QUERY_INSTRUCTION = "Given a question about past agent trajectories, retrieve relevant memory entries that help answer it."

# ---------------------------------------------------------------- prompts (SELF-INDEX Tables 6-10, verbatim)
P_DIAG = """You maintain the search index entry of the document below. The entry is a set of KEYS: texts the search engine matches against user queries. A query retrieves the document through whichever of its keys matches best. The retrieval feedback below shows how the current keys behave for observed queries.
DOCUMENT:
<<<DOC START>>>
{doc}
<<<DOC END>>>
CURRENT KEYS (numbered):
{keys}
RETRIEVAL FEEDBACK (query texts, retrieved keys and their scores, and cross-document co-retrieval frequencies):
{feedback}
Read the document and examine its current keys together with the retrieval feedback. Determine whether the current representations have any shortfalls that warrant revision. Multiple documents may satisfy the same query, so co-retrieval alone does not establish a shortfall.
Identify the current key responsible for each shortfall. If the shortfall concerns insufficient coverage by the key set as a whole rather than a particular key, use "key set". Explain what the current representations fail to express or distinguish. Then state concretely what a revision of the key set should change. Do not write the revised keys.
If no shortfall is supported, return an empty diagnoses list and an empty revision string.
Answer JSON only:
{{"diagnoses": [
{{"key": <number or "key_set">,
"cause": "<why it falls short>"}}
],
"revision": "<what to change>"}}"""

P_REVISE = """You maintain the search index entry of the document below: a set of KEYS, short texts the search engine matches against user queries. A query retrieves the document through whichever key matches best.
DOCUMENT:
<<<DOC START>>>
{doc}
<<<DOC END>>>
CURRENT KEYS (numbered; a diagnosed key is marked with what falls short):
{keys}
SUGGESTED CHANGES FROM SELF-DIAGNOSIS:
{diagnosis}
COMPETING KEYS (index entries of other documents retrieved for the same queries; use these to identify what the current document provides that its keys should distinguish more clearly):
{neighbors}
Revise the key set as a whole. Keep the keys that still serve the document, drop or rewrite the diagnosed ones, and write at most {k} new or rewritten keys so that anyone who needs what this document provides can find it, whatever they already know and however they would ask. Write each key in the words such a searcher would use, not in the document's words. Decide the form and length of each key yourself, from the document and from how people would look for it. Each key must serve a need, a situation, or a way of asking that the other keys do not. Do not restate or copy the document; do not repeat a kept key.
Every key you write must satisfy three requirements:
- FAITHFUL: it describes only knowledge this document actually contains or resolves; nothing invented, nothing the document does not support.
- SPECIFIC: it captures what is specific to THIS document, not content broadly shared with other documents in the corpus.
- SEPARATED: it reduces ambiguity with the competing keys by emphasizing distinctions supported by this document. A shared information need is acceptable when this document also addresses it.
Answer JSON only:
{{"keep": [<numbers of keys to keep>],
"revised": ["<key>", ...]}}"""

JUDGE_RULES = """- First identify the information need behind it. A search may be a question, a how-to, a bare keyword phrase, a claim to verify, or a scenario with a persona ("I'm a nurse who ..."). Judge the need, not the wording or the framing. Searchers use everyday words where the document uses technical ones, and that is fine.
- The need is RESOLVED (YES) when the document itself contains what the searcher needs: the answer, the explanation or mechanism, the procedure or technique, or the definition. It is also resolved when the document states a fact or principle from which the answer follows with ordinary reasoning. A claim is resolved when the document states, supports, or refutes it. A keyword or topic search is resolved when the document substantively covers that topic. A search for a source, study, or overview is resolved when this document is that source.
- The need is NOT resolved (NO) when the document only shares the topic or mentions it in passing, when the need turns on a specific detail the document lacks (a reason why, a comparison, a number, a step-by-step procedure, a result or outcome), or when the document is only a title, heading, citation entry, reference list, table fragment, or figure caption that names the subject without containing the content."""

P_FAITH = """You are auditing a search index. Below is one document and a numbered list of SEARCHES written so that this document can be found. For each search rate how well THIS document would resolve it for the user who typed it.
Document:
<<<DOC START>>>
{doc}
<<<DOC END>>>
Searches:
{views}
How to judge each search:
""" + JUDGE_RULES + """
- Judge each search on its own. The verdict on one search must not influence another.
Scale:
0 the document has nothing to do with the need
1 related topic but the need is not met
2 the document meets the need, even if the answer is partial, implicit, or amid other material
3 the document is dedicated to the need and contains the exact answer
Reply format: one line per search, in order: the number, a colon, at most twelve words naming the need and what the document offers for it, then => and the score. Nothing else.
Example: 3: asks why, document lists causes only => 1."""

P_ANSWER = """You are auditing a search index. Below is one document and one SEARCH written so that this document can be found. Rate how well THIS document would resolve it for the user who typed it.
Document:
<<<DOC START>>>
{doc}
<<<DOC END>>>
Search:
{query}
How to judge the search:
""" + JUDGE_RULES + """
Scale:
0 the document has nothing to do with the need
1 related topic but the need is not met
2 the document meets the need, even if the answer is partial, implicit, or amid other material
3 the document is dedicated to the need and contains the exact answer
Reply format: one line: the score, then a dash and at most ten words of reason. Nothing else."""

P_ABSTRACT = """Read the document and state, in plain language, the underlying problem, phenomenon or question that this document resolves for a reader. Describe it the way a person who has NOT read the document would experience or wonder about it: what they observe, want, or are confused by. Do not name the document's specialized terms, entities or titles.
Document:
<<<DOC START>>>
{doc}
<<<DOC END>>>
Answer JSON only:
{{"problem": "..."}}"""

P_ABSTRACT_UI = """Read the recorded web page below and list, in plain language, the INTERFACE FACTS a person looking at this page could see: which site section and page it is; which buttons, links, tabs, menus, columns and filter or sort options are visible and what they are called; which fields or labels are shown and what values or states they have (checked, selected, read-only, required); any banner, notice or message text and where it sits; what is in the sidebars or header. Describe what is visible, not what task someone is doing. Do not copy raw markup.
Document:
<<<DOC START>>>
{doc}
<<<DOC END>>>
Answer JSON only:
{{"facts": ["...", "..."]}}"""

P_WRITEQ_UI = """You are an assistant that has used this website before and now needs to look up what a particular page shows. You cannot see the page. From the list of interface facts below, write {m} different short search queries, each asking about a DIFFERENT visible element or property: whether a named control is present, what a label or field says, what a banner or message reads, which options a sort or filter offers, where an element sits, or what the navigation path to the page is. Write each query as a compact noun phrase naming the site area, the page, the element and the property, the way a search box query would be written, not as a full question and not with the facts' exact wording.
Interface facts:
{facts}
Answer JSON only:
{{"queries": ["...", "..."]}}"""

P_WRITEQ = """You are a person in the situation below. You have NOT read any document about it and you do not know its technical vocabulary. Write {m} different questions you would post on a Q&A forum: your concrete observation or goal, what puzzles you, and what you have tried or assumed. Write in everyday language, and make each question a different need or a different way of asking.
Situation:
{problem}
Answer JSON only:
{{"queries": ["...", "..."]}}"""


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- data
def load_pool(dom: str):
    """ids (pool order), doc texts for the LLM, struct heads for neighbour display."""
    struct = {r["entry_id"]: r["struct"] for r in map(json.loads, open(V2 / "keys" / dom / "struct.jsonl"))}
    ids, docs, heads = [], [], []
    with open(E1 / f"index/agentrunbook_r_{dom}_small/memory_state/pools/raw_state_pool.jsonl") as fp:
        for line in fp:
            e = json.loads(line)
            st = struct.get(e["entry_id"], "")[:STRUCT_CHARS]
            ids.append(e["entry_id"])
            docs.append("\n".join([f"Task goal: {e['goal'].strip()[:1500]}",
                                   f"Actions at this step:\n{e['slice_action_sequence'].strip()[:2000]}",
                                   f"URLs: {' '.join(e['slice_urls'])[:1000]}",
                                   f"Page structure (named elements):\n{st or '(none)'}",
                                   f"Page content (accessibility tree, may be cut off):\n{e['slice_axtree_text'][:DOC_AXTREE_CHARS]}"]))
            heads.append((e["goal"].strip()[:120] + " | " + (st or e["slice_axtree_text"][:NEIGH_CHARS])[:NEIGH_CHARS]).replace("\n", " "))
            del e
    return ids, docs, heads


class LLM:
    def __init__(self):
        from openai import OpenAI
        self.c = OpenAI(base_url=os.environ["READER_BASE_URL"], api_key="EMPTY", timeout=900)
        self.model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
        self.calls, self.in_tok, self.out_tok = 0, 0, 0

    def ask(self, prompt: str, max_tokens: int = 1200) -> str:
        for attempt in range(3):
            try:
                r = self.c.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}],
                                                   temperature=0.0, max_tokens=max_tokens,
                                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                self.calls += 1; self.in_tok += r.usage.prompt_tokens; self.out_tok += r.usage.completion_tokens
                return (r.choices[0].message.content or "").strip()
            except Exception as ex:  # noqa: BLE001
                if attempt == 2:
                    log(f"LLM error (giving up): {ex!r}")
                    return ""
                time.sleep(5)
        return ""

    def pmap(self, fn, items, workers):
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(fn, items))


def parse_json(s: str):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    for cand in (m.group(0), m.group(0).replace("\n", " ")):
        try:
            return json.loads(cand)
        except Exception:  # noqa: BLE001
            continue
    return None


class Embedder:
    """Document-form (no instruction, like the official raw-state embedding) and query-form vectors, cached."""

    def __init__(self, dom: str):
        from openai import OpenAI
        self.c = OpenAI(base_url=os.environ["LME_EMBEDDING_BASE_URL"], api_key="EMPTY", timeout=600)
        self.model = os.environ.get("LME_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
        self.path = V2 / TAG / dom / "keyvec.npz"
        self.doc: dict[str, np.ndarray] = {}
        self.qry: dict[str, np.ndarray] = {}
        if self.path.exists():
            z = np.load(self.path, allow_pickle=True)
            texts = list(z["texts"])
            self.doc = dict(zip(texts, z["doc"]))
            self.qry = dict(zip(texts, z["qry"]))

    def _embed(self, texts: list[str], as_query: bool) -> np.ndarray:
        prepared = [f"Instruct: {QUERY_INSTRUCTION}\nQuery:{t.strip()}" if as_query else t for t in texts]
        out = []
        for k in range(0, len(prepared), 32):
            resp = self.c.embeddings.create(model=self.model, input=[p[:6000] for p in prepared[k:k + 32]])
            v = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)

    def ensure(self, texts: list[str]) -> None:
        new = [t for t in dict.fromkeys(texts) if t not in self.doc]
        if not new:
            return
        D, Q = self._embed(new, False), self._embed(new, True)
        for t, d, q in zip(new, D, Q):
            self.doc[t], self.qry[t] = d, q

    def queries(self, texts: list[str]) -> np.ndarray:
        """Query-form vectors for simulated queries (not cached with keys)."""
        return self._embed(texts, True)

    def save(self) -> None:
        texts = list(self.doc)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not texts:
            return
        np.savez(self.path, texts=np.asarray(texts, dtype=object), doc=np.stack([self.doc[t] for t in texts]),
                 qry=np.stack([self.qry[t] for t in texts]))


# ---------------------------------------------------------------- index
class Index:
    def __init__(self, dom: str, ids: list[str]):
        self.dom, self.ids = dom, ids
        self.pos = {e: i for i, e in enumerate(ids)}
        self.E = np.load(E1 / f"index/agentrunbook_r_{dom}_small/memory_state/embeddings/raw_state.npy").astype(np.float32)
        self.path = V2 / TAG / dom / "keys.json"
        self.keys: dict[str, list[str]] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.keys, ensure_ascii=False, indent=0))

    def key_table(self, emb: Embedder):
        """Flat key list: (doc index, key text or None for the original), doc-form matrix."""
        owners, texts, rows = [], [], []
        for i, e in enumerate(self.ids):
            owners.append(i); texts.append(None); rows.append(self.E[i])
            for k in self.keys.get(e, []):
                owners.append(i); texts.append(k); rows.append(emb.doc[k])
        return np.asarray(owners), texts, np.stack(rows)

    def doc_scores(self, Q: np.ndarray, emb: Embedder) -> np.ndarray:
        """max over keys, (n_queries, n_docs)."""
        owners, _, M = self.key_table(emb)
        S = Q @ M.T
        out = np.full((Q.shape[0], len(self.ids)), -np.inf, np.float32)
        for j in range(M.shape[0]):
            np.maximum(out[:, owners[j]], S[:, j], out=out[:, owners[j]])
        return out


# ---------------------------------------------------------------- simulator (Algorithm 3)
def jaccard(a: str, b: str) -> float:
    A, B = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(A & B) / max(1, len(A | B))


def simulate(dom: str, llm: LLM, ids, docs, rng: random.Random, used: list[str], workers: int, log_rec: dict) -> list[tuple[str, int]]:
    sample = rng.sample(range(len(ids)), min(SIM_DOCS, len(ids)))

    def abstract(i):
        if SIM == "ui":
            j = parse_json(llm.ask(P_ABSTRACT_UI.format(doc=docs[i]), 700)) or {}
            facts = [str(x).strip() for x in j.get("facts", []) if str(x).strip()]
            return "\n".join(f"- {f}" for f in facts[:20])
        j = parse_json(llm.ask(P_ABSTRACT.format(doc=docs[i]), 400)) or {}
        return str(j.get("problem", "")).strip()
    problems = llm.pmap(abstract, sample, workers)

    def write(p):
        if not p:
            return []
        if SIM == "ui":
            j = parse_json(llm.ask(P_WRITEQ_UI.format(m=SIM_M, facts=p), 600)) or {}
        else:
            j = parse_json(llm.ask(P_WRITEQ.format(m=SIM_M, problem=p), 600)) or {}
        return [str(q).strip() for q in j.get("queries", []) if str(q).strip()][:SIM_M]
    cands = llm.pmap(write, problems, workers)
    pairs = [(q, i) for i, qs in zip(sample, cands) for q in qs]

    def answerable(pq):
        q, i = pq
        s = llm.ask(P_ANSWER.format(doc=docs[i], query=q), 60)
        m = re.match(r"\s*(\d)", s)
        return int(m.group(1)) if m else -1
    scores = llm.pmap(answerable, pairs, workers)
    accepted, pool = [], list(used)
    for (q, i), sc in zip(pairs, scores):
        if sc >= JUDGE_MIN and all(jaccard(q, q2) < JACCARD_T for q2 in pool):
            accepted.append((q, i)); pool.append(q)
    rng.shuffle(accepted)
    accepted = accepted[:BATCH_Q]
    log_rec["sim"] = {"sampled_docs": len(sample), "candidates": len(pairs), "answerable": sum(s >= JUDGE_MIN for s in scores),
                      "accepted": len(accepted), "examples": [q for q, _ in accepted[:5]]}
    return accepted


# ---------------------------------------------------------------- one iteration (Algorithms 1-2)
def iterate(dom: str, t: int, llm: LLM, emb: Embedder, idx: Index, ids, docs, heads, rng, workers: int) -> None:
    rec = {"iter": t}
    qpath = V2 / TAG / dom / "queries.jsonl"
    used = [json.loads(l)["query"] for l in open(qpath)] if qpath.exists() else []
    Qs = simulate(dom, llm, ids, docs, rng, used, workers, rec)
    if not Qs:
        log(f"iter {t}: no accepted queries"); return
    with open(qpath, "a") as fp:
        for q, i in Qs:
            fp.write(json.dumps({"iter": t, "query": q, "source": ids[i]}) + "\n")
    qtexts = [q for q, _ in Qs]
    Qv = emb.queries(qtexts)
    # key-level retrieval on the iteration-start snapshot
    owners, ktexts, M = idx.key_table(emb)
    S = Qv @ M.T
    topk = np.argsort(-S, axis=1)[:, :TOP_K]
    prof: dict[int, Counter] = defaultdict(Counter)       # key j -> Counter of other-doc keys co-retrieved
    hits: dict[int, list] = defaultdict(list)             # doc i -> [(query idx, [(key j, score)])]
    for qi in range(len(qtexts)):
        ks = [int(j) for j in topk[qi]]
        for j in ks:
            for j2 in ks:
                if owners[j2] != owners[j]:
                    prof[j][j2] += 1
        bydoc = defaultdict(list)
        for j in ks:
            bydoc[int(owners[j])].append((j, float(S[qi, j])))
        for i, lst in bydoc.items():
            hits[i].append((qi, lst))
    rec["retrieved_docs"] = len(hits)

    def key_label(j):
        return "(original page text)" if ktexts[j] is None else ktexts[j]

    def neigh_label(j):
        return heads[owners[j]] if ktexts[j] is None else ktexts[j]

    # -------- diagnose
    targets = sorted(hits)
    doc_js = {i: [j for j in np.nonzero(owners == i)[0].tolist()] for i in targets}

    def diagnose(i):
        js = doc_js[i]
        keys_txt = "\n".join(f"{n}. {key_label(j)}" for n, j in enumerate(js))
        fb = []
        for qi, lst in hits[i][:8]:
            fb.append(f"- query: {qtexts[qi]!r}; retrieved keys of this document: " +
                      ", ".join(f"key {js.index(j)} (score {s:.3f})" for j, s in lst))
        for n, j in enumerate(js):
            if prof[j]:
                top = prof[j].most_common(6)
                fb.append(f"- key {n} co-retrieved with other documents' keys: " + "; ".join(f"{neigh_label(j2)[:160]!r} x{c}" for j2, c in top))
        out = parse_json(llm.ask(P_DIAG.format(doc=docs[i], keys=keys_txt, feedback="\n".join(fb)), 700)) or {}
        d = out.get("diagnoses") or []
        return (i, d, str(out.get("revision", "")).strip()) if d and out.get("revision") else (i, [], "")
    diags = llm.pmap(diagnose, targets, workers)
    todo = [(i, d, r) for i, d, r in diags if d]
    rec["diagnosed"] = len(todo)

    # -------- revise
    def revise(item):
        i, d, guidance = item
        js = doc_js[i]
        flagged = set()
        for x in d:
            k = x.get("key")
            if k == "key_set":
                flagged.update(range(len(js)))
            elif isinstance(k, int) and 0 <= k < len(js):
                flagged.add(k)
            elif isinstance(k, str) and k.isdigit() and int(k) < len(js):
                flagged.add(int(k))
        if not flagged:
            flagged.update(range(len(js)))
        comp = Counter()
        for n in flagged:
            comp.update(prof[js[n]])
        comp_js = [j for j, _ in comp.most_common(12)]
        marks = {}
        for x in d:
            k = x.get("key")
            n = int(k) if isinstance(k, (int, str)) and str(k).isdigit() else None
            if n is not None:
                marks[n] = x.get("cause", "")
        gen = js[1:]  # generated keys only are numbered in the revision prompt (original is fixed)
        keys_txt = "\n".join(f"{n}. {ktexts[j]}" + (f"  [diagnosed: {marks[n + 1]}]" if (n + 1) in marks else "")
                             for n, j in enumerate(gen)) or "(no generated keys yet)"
        diag_txt = guidance + ("" if 0 not in marks else f"\n(original page text key: {marks[0]})")
        neigh = "\n".join(f"- {neigh_label(j)[:NEIGH_CHARS]}" for j in comp_js) or "(none)"
        k_new = MAX_GEN
        out = parse_json(llm.ask(P_REVISE.format(doc=docs[i], keys=keys_txt, diagnosis=diag_txt, neighbors=neigh, k=k_new), 900)) or {}
        keep = [n for n in out.get("keep", []) if isinstance(n, int) and 0 <= n < len(gen)]
        new = [str(s).strip() for s in out.get("revised", []) if str(s).strip()]
        kept = [ktexts[gen[n]] for n in keep]
        new = [s for s in new if s not in kept][: max(0, MAX_GEN - len(kept))]
        return i, kept, new, comp_js
    props = llm.pmap(revise, todo, workers)
    props = [p for p in props if p[2]]
    rec["proposed"] = len(props)
    emb.ensure([k for _, kept, new, _ in props for k in kept + new])

    # -------- validate: faithfulness (LLM), specificity and separation (embedding)
    def faith(p):
        i, kept, new, _ = p
        cand = kept + new
        views = "\n".join(f"{n + 1}. {k}" for n, k in enumerate(cand))
        s = llm.ask(P_FAITH.format(doc=docs[i], views=views), 40 * len(cand) + 50)
        sc = {}
        for line in s.splitlines():
            m = re.match(r"\s*(\d+)\s*:.*=>\s*(\d)", line)
            if m:
                sc[int(m.group(1))] = int(m.group(2))
        return [k for n, k in enumerate(cand) if sc.get(n + 1, -1) >= JUDGE_MIN]
    faithful = llm.pmap(faith, props, workers)
    n_f = n_spec = n_sep = n_upd = 0
    examples = []
    for (i, kept, new, comp_js), ok_f in zip(props, faithful):
        n_f += len(ok_f)
        passing = []
        cur_js = doc_js[i]
        # separation baseline P_d: current key set (incl. original) vs competing keys
        if comp_js:
            C = np.stack([M[j] for j in comp_js])                      # competing keys, doc form
            cur_q = np.stack([(emb.qry[ktexts[j]] if ktexts[j] is not None else idx.E[i]) for j in cur_js])
            P_d = float((cur_q @ C.T).max())
        for k in ok_f:
            qv = emb.qry[k]
            # specificity: key as query against all original texts; source doc must be within TOP_K
            s_all = idx.E @ qv
            if int((s_all >= s_all[i]).sum() - 1) >= TOP_K:
                continue
            n_spec += 1
            if comp_js and float((C @ qv).max()) >= P_d:
                continue
            n_sep += 1
            passing.append(k)
        if any(k in new for k in passing):
            idx.keys[ids[i]] = passing[:MAX_GEN]
            n_upd += 1
            if len(examples) < 6:
                examples.append({"entry": ids[i], "goal": heads[i][:100], "keys": passing[:MAX_GEN]})
    rec.update({"faithful_keys": n_f, "specific_keys": n_spec, "separated_keys": n_sep, "docs_updated": n_upd,
                "docs_with_keys": sum(1 for e in ids if idx.keys.get(e)), "total_gen_keys": sum(len(v) for v in idx.keys.values()),
                "examples": examples, "llm_calls": llm.calls, "llm_in_tok": llm.in_tok, "llm_out_tok": llm.out_tok})
    idx.save(); emb.save()
    (V2 / TAG / dom / f"iter_{t}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    log(f"iter {t} {dom}: q={len(qtexts)} retrieved_docs={len(hits)} diagnosed={len(todo)} proposed={len(props)} "
        f"faithful={n_f} specific={n_spec} separated={n_sep} updated={n_upd} docs_with_keys={rec['docs_with_keys']} keys={rec['total_gen_keys']}")


def do_iterate(dom: str, iters: int, workers: int, seed: int) -> None:
    (V2 / TAG / dom).mkdir(parents=True, exist_ok=True)
    ids, docs, heads = load_pool(dom)
    llm, emb, idx = LLM(), Embedder(dom), Index(dom, ids)
    emb.ensure([k for v in idx.keys.values() for k in v])
    done = sorted(int(p.stem.split("_")[1]) for p in (V2 / TAG / dom).glob("iter_*.json")) if (V2 / TAG / dom).exists() else []
    start = (done[-1] + 1) if done else 0
    rng = random.Random(seed + start)
    for t in range(start, start + iters):
        iterate(dom, t, llm, emb, idx, ids, docs, heads, rng, workers)


def do_score(dom: str) -> None:
    """SI = max(V, keys) and SIk = keys only, for the frozen E2 queries, in the E2 matrix layout."""
    from e1.retrieval.base import QueryVectors
    ids, _, _ = load_pool(dom)
    emb, idx = Embedder(dom), Index(dom, ids)
    emb.ensure([k for v in idx.keys.values() for k in v]); emb.save()
    d = V2 / "scores" / dom
    rows = json.loads((d / "queries.json").read_text())
    qv = QueryVectors(dom)
    Q = np.stack([qv.get(r["text"]) for r in rows]).astype(np.float32)
    V = np.asarray(np.load(d / "V.npy"), np.float32)
    owners, ktexts, M = idx.key_table(emb)
    gen = [j for j in range(len(owners)) if ktexts[j] is not None]
    SIk = np.full_like(V, -np.inf)
    if gen:
        S = Q @ M[gen].T
        for n, j in enumerate(gen):
            np.maximum(SIk[:, owners[j]], S[:, n], out=SIk[:, owners[j]])
    np.save(d / f"{MAT}k.npy", SIk)
    np.save(d / f"{MAT}.npy", np.maximum(V, SIk))
    log(f"score {dom}: {len(gen)} generated keys over {len(set(owners[gen].tolist()) if gen else [])} docs; "
        f"SI/SIk {V.shape} -> {d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["iterate", "score"])
    ap.add_argument("--domains", default="web")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1102)
    a = ap.parse_args()
    for dom in a.domains.split(","):
        if a.stage == "iterate":
            do_iterate(dom, a.iters, a.workers, a.seed)
        else:
            do_score(dom)


if __name__ == "__main__":
    main()
