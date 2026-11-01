#!/usr/bin/env python3
"""TRAD v2: step-level retrieval keys for the AgentRunbook-R raw-state pool (retrieval only, no reader).

Pre-registration: CP5101/experiments/trad-v2-plan-2026-11-02.md (written before any v2 output).

Methods (score matrices follow the E2 layout so retrieve.py's ScoreBank / select work unchanged):
  K0  deterministic structure key from the FULL slice AXTree (named roles, chrome lines dropped)
      + goal + slice actions + urls; BM25 (camel)
  K1  LLM step key (TRAD pseudo-golden-thought analogue): Qwen3.5-9B, no thinking, T=0, <= 300 tokens,
      from goal + slice actions + urls + K0 structure + the first HEAD_CHARS of the slice AXTree;
      embedded with Qwen3-Embedding-8B (documents, no instruction), cosine with the E2 query vectors

Work dir ~/e1/v2work (E2 outputs are read-only; V and T:camel are symlinked in):
  keys/<dom>/struct.jsonl   K0 texts (entry order = pool order)
  keys/<dom>/llm.jsonl      K1 texts (append-only, resumable; one row per entry_id)
  keys/<dom>/llm_vectors.npy
  scores/<dom>/{queries.json,V.npy,T_camel.npy,K0.npy,K1.npy}

Stages:
  struct            CPU
  write             GPU, needs the reader server ($READER_BASE_URL)
  embed             GPU, needs the embedding server ($LME_EMBEDDING_BASE_URL)
  score             CPU
  eval --evidence F CPU; F = the committed evidence rules (experiments/trad-v2-evidence-2026-11-02.json)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
E1 = Path(os.environ.get("E1", "~/e1")).expanduser()
V2 = Path(os.environ.get("V2WORK", str(E1 / "v2work")))
os.environ.setdefault("E2WORK", str(V2))  # retrieve.ScoreBank reads $E2WORK/scores
DOMS = ("web", "enterprise")

HEAD_CHARS = 24000
STRUCT_CHARS = 6000
CHROME_DF = 0.3
ROLE_ORDER = ["RootWebArea", "heading", "tab", "columnheader", "LabelText", "textbox", "combobox", "checkbox",
              "radio", "searchbox", "button", "option", "menuitem", "alert", "status", "link"]
ROLE_RE = re.compile(r"^\s*(?:\[\w+\]\s*)?(" + "|".join(ROLE_ORDER) + r")\s+'([^']{1,120})'")
STATE_RE = re.compile(r"(?m)^State \d+\s*$")

PROMPT = """You are indexing one step of a recorded web-agent trajectory so that it can be found later by questions about how this website looks and works.

Task goal of the trajectory: {goal}
Actions in this part of the trajectory:
{actions}
URLs: {urls}

Page structure (named elements found anywhere on these pages, site-wide navigation removed):
{struct}

Page content (accessibility tree, may be cut off):
{axtree}

Write a retrieval key of at most 120 words, exactly these four lines:
Page: <which page this is: site section, page title, record or list name>
Shows: <specific entity / record / item names, grid or list names, column names, tab names, field labels with their values or states (read-only, mandatory, selected, checked), banners or messages>
Controls: <buttons, links, dropdown / sort / filter options that are visible>
Step: <what the agent did here and what changed on the page>
Use only what is on the page. Do not guess."""


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pool(dom: str):
    with open(E1 / f"index/agentrunbook_r_{dom}_small/memory_state/pools/raw_state_pool.jsonl") as fp:
        for line in fp:
            yield json.loads(line)


def role_items(axtree: str) -> list[tuple[str, str]]:
    out = []
    for line in axtree.splitlines():
        m = ROLE_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


# ---------------------------------------------------------------- K0
def do_struct(dom: str) -> None:
    rows = []
    df = Counter()
    for e in pool(dom):
        items = list(dict.fromkeys(role_items(e["slice_axtree_text"])))
        rows.append(({k: e[k] for k in ("entry_id", "goal", "slice_action_sequence", "slice_urls")}, items))
        df.update(set(items))
        del e
    n = len(rows)
    chrome = {k for k, c in df.items() if c >= CHROME_DF * n}
    out = V2 / "keys" / dom
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "struct.jsonl", "w") as fp:
        for e, items in rows:
            keep = sorted((x for x in items if x not in chrome), key=lambda x: ROLE_ORDER.index(x[0]))
            s, used = [], 0
            for role, name in keep:
                t = f"{role}: {name}"
                if used + len(t) + 1 > STRUCT_CHARS:
                    break
                s.append(t); used += len(t) + 1
            fp.write(json.dumps({"entry_id": e["entry_id"], "struct": "\n".join(s),
                                 "text": "\n".join([e["goal"], e["slice_action_sequence"], " ".join(e["slice_urls"]), "\n".join(s)])}) + "\n")
    log(f"struct {dom}: {n} entries, {len(chrome)} chrome items dropped")


# ---------------------------------------------------------------- K1 write
def do_write(dom: str, workers: int, limit: int | None) -> None:
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["READER_BASE_URL"], api_key="EMPTY", timeout=600)
    model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
    struct = {r["entry_id"]: r["struct"] for r in map(json.loads, open(V2 / "keys" / dom / "struct.jsonl"))}
    path = V2 / "keys" / dom / "llm.jsonl"
    done = {json.loads(l)["entry_id"] for l in open(path)} if path.exists() else set()
    todo = [e for e in pool(dom) if e["entry_id"] not in done]
    if limit:
        todo = todo[:limit]
    log(f"write {dom}: {len(done)} done, {len(todo)} to write")
    t0 = time.perf_counter()

    def one(e):
        msg = PROMPT.format(goal=e["goal"].strip()[:1500], actions=e["slice_action_sequence"].strip()[:3000],
                            urls=" ".join(e["slice_urls"])[:1500], struct=struct[e["entry_id"]] or "(none)",
                            axtree=e["slice_axtree_text"][:HEAD_CHARS])
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": msg}], temperature=0.0,
                                           max_tokens=300, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return {"entry_id": e["entry_id"], "key": (r.choices[0].message.content or "").strip(),
                "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens,
                "finish_reason": r.choices[0].finish_reason}

    with open(path, "a") as fp, ThreadPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            fp.write(json.dumps(rec, ensure_ascii=True) + "\n"); fp.flush()
            if i % 100 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                log(f"write {dom}: {i}/{len(todo)} in {el:.0f}s (eta {el / i * (len(todo) - i):.0f}s)")


# ---------------------------------------------------------------- K1 embed
def do_embed(dom: str, batch: int, workers: int) -> None:
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["LME_EMBEDDING_BASE_URL"], api_key="EMPTY", timeout=600)
    model = os.environ.get("LME_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
    from e1.pipeline.retrieve import entry_ids
    ids = entry_ids(dom)
    keys = {}
    for l in open(V2 / "keys" / dom / "llm.jsonl"):
        r = json.loads(l); keys[r["entry_id"]] = r["key"]
    missing = [e for e in ids if not keys.get(e)]
    assert not missing, f"{len(missing)} entries without a key, e.g. {missing[:3]}"
    texts = [keys[e] for e in ids]

    def one(k):
        resp = client.embeddings.create(model=model, input=texts[k * batch:(k + 1) * batch])
        v = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        return k, v / np.linalg.norm(v, axis=1, keepdims=True)
    n = (len(texts) + batch - 1) // batch
    out = [None] * n
    with ThreadPoolExecutor(workers) as ex:
        for k, v in ex.map(one, range(n)):
            out[k] = v
    V = np.concatenate(out)
    np.save(V2 / "keys" / dom / "llm_vectors.npy", V)
    log(f"embed {dom}: {V.shape}")


# ---------------------------------------------------------------- score
def do_score(dom: str) -> None:
    from e1.retrieval.base import QueryVectors
    from e1.retrieval.bm25 import BM25, tokenize
    from e1.pipeline.retrieve import entry_ids
    e2 = E1 / "e2work/scores" / dom
    d = V2 / "scores" / dom
    d.mkdir(parents=True, exist_ok=True)
    for f in ("queries.json", "V.npy", "T_camel.npy"):
        if not (d / f).exists():
            (d / f).symlink_to(e2 / f)
    rows = json.loads((d / "queries.json").read_text())
    ids = entry_ids(dom)
    st = [json.loads(l) for l in open(V2 / "keys" / dom / "struct.jsonl")]
    assert [r["entry_id"] for r in st] == ids
    bm = BM25.from_texts([r["text"] for r in st], tokenizer="camel")
    np.save(d / "K0.npy", np.stack([bm.scores_terms(tokenize(r["text"], "camel")) for r in rows]).astype(np.float32))
    kv = V2 / "keys" / dom / "llm_vectors.npy"
    if kv.exists():
        K = np.load(kv)
        assert K.shape[0] == len(ids)
        qv = QueryVectors(dom)
        Q = np.stack([qv.get(r["text"]) for r in rows]).astype(np.float32)
        np.save(d / "K1.npy", (Q @ K.T).astype(np.float32))
    log(f"score {dom}: K0{' K1' if kv.exists() else ''} for {len(rows)} queries x {len(ids)} entries")


# ---------------------------------------------------------------- eval
def state_texts(e: dict) -> list[str]:
    t = e["slice_axtree_text"]
    parts = [p for p in STATE_RE.split(t) if p.strip()]
    return [p.lower() for p in parts] or [t.lower()]


def matches(rule: dict, e: dict, states: list[str]) -> bool:
    urls = " ".join(e["slice_urls"]).lower()
    goal = e["goal"].lower()
    if rule.get("url") and not re.search(rule["url"], urls):
        return False
    if rule.get("goal") and not re.search(rule["goal"], goal):
        return False
    need = [x.lower() for x in rule.get("all", [])]
    anyof = [x.lower() for x in rule.get("any", [])]
    none = [x.lower() for x in rule.get("none", [])]
    for s in states:
        if all(x in s for x in need) and (not anyof or any(x in s for x in anyof)) and not any(x in s for x in none):
            return True
    return False


def evidence_sets(rules: dict) -> dict:
    """{qid: {group: set(entry_id)}} over the whole pool."""
    out = {q: {g: set() for g in r["groups"]} for q, r in rules.items()}
    for dom in DOMS:
        R = {q: r for q, r in rules.items() if r["domain"] == dom}
        if not R:
            continue
        for e in pool(dom):
            st = None
            for q, r in R.items():
                for g, rule in r["groups"].items():
                    if st is None:
                        st = state_texts(e)
                    if matches(rule, e, st):
                        out[q][g].add(e["entry_id"])
    return out


def do_eval(evidence: Path, variants: list[str], out: Path) -> None:
    from e1.pipeline.retrieve import ScoreBank, do_select, entry_ids, meta
    from e1.analysis.retrieval_eval import Content, label_question
    rules = {k: v for k, v in json.loads(evidence.read_text())["questions"].items()}
    ev = evidence_sets(rules)
    sel = {}
    for dom in DOMS:
        bank, ids = ScoreBank(dom), entry_ids(dom)
        for v in variants:
            for r in do_select(dom, v, "frozen", None, bank=bank, ids=ids, sim_variant=None):
                sel[(v, r["question_id"])] = r
    full = {q for (_, q) in sel}
    res = {"evidence_counts": {q: {g: len(s) for g, s in G.items()} for q, G in ev.items()}, "variants": variants}
    # target set
    per_q = {}
    for q8, G in ev.items():
        q = next(x for x in full if x.startswith(q8))
        row = {"in_pool": all(len(s) > 0 for s in G.values())}
        for v in variants:
            blocks = {b["entry_id"] for b in sel[(v, q)]["raw_blocks"]}
            row[v] = {"covered": all(bool(blocks & s) for s in G.values()),
                      "groups": {g: bool(blocks & s) for g, s in G.items()}}
            # best per-query rank of any evidence entry (diagnostic)
        per_q[q8] = row
    res["target"] = per_q
    ref = variants[0]
    summ = {}
    E = [q for q, r in per_q.items() if r["in_pool"] and not r[ref]["covered"]]
    for v in variants[1:]:
        gain = [q for q in E if per_q[q][v]["covered"]]
        loss = [q for q, r in per_q.items() if r[ref]["covered"] and not r[v]["covered"]]
        summ[v] = {"eligible": len(E), "gain": len(gain), "gain_ids": gain, "loss": len(loss), "loss_ids": loss}
    res["target_summary"] = {"ref": ref, "not_in_pool": [q for q, r in per_q.items() if not r["in_pool"]],
                             "already_covered_by_ref": [q for q, r in per_q.items() if r[ref]["covered"]], **summ}
    # guardrail: E2 answer-phrase proxy + merged-block token counts
    guard = {}
    for dom in DOMS:
        M = meta(dom)
        C = Content(dom)
        qs = sorted(q for (vv, q) in sel if vv == ref and q in M)
        need = {b["entry_id"] for (vv, q), r in sel.items() if q in M for b in r["raw_blocks"]}
        C.load(need)
        lab = {v: {q: label_question(M[q], [b["entry_id"] for b in sel[(v, q)]["raw_blocks"]], C) for q in qs} for v in variants}
        for v in variants[1:]:
            el = [q for q in qs if "ans" in lab[ref][q]]
            g = sum(lab[v][q]["ans"] and not lab[ref][q]["ans"] for q in el)
            l = sum(lab[ref][q]["ans"] and not lab[v][q]["ans"] for q in el)
            guard.setdefault(v, {})[dom] = {"eligible": len(el), "ref_cov": sum(lab[ref][q]["ans"] for q in el),
                                             "cov": sum(lab[v][q]["ans"] for q in el), "gain": g, "loss": l}
    res["guard_ans"] = guard
    # merged-block size (characters of the full slice text as a length proxy; tokens ~ chars / 4)
    size = {}
    for dom in DOMS:
        L = {}
        for e in pool(dom):
            L[e["entry_id"]] = len(e["slice_axtree_text"])
        for v in variants:
            tot = [sum(L.get(b["entry_id"], 0) for b in r["raw_blocks"]) for (vv, q), r in sel.items() if vv == v and r["raw_blocks"] and r["raw_blocks"][0]["entry_id"] in L]
            size.setdefault(v, {})[dom] = statistics.median(tot) if tot else None
    res["median_block_chars"] = size
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=list))
    print(json.dumps({"target_summary": res["target_summary"], "guard_ans": guard, "median_block_chars": size}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["struct", "write", "embed", "score", "eval"])
    ap.add_argument("--domains", default="web,enterprise")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--variants", default="V,rrf:10:V+K1,K1,K0,rrf:10:V+K0,rrf:10:V+K0+K1,rrf:10:V+T:camel")
    ap.add_argument("--out", default=str(V2 / "reports/eval.json"))
    a = ap.parse_args()
    if a.stage == "eval":
        do_eval(Path(a.evidence).expanduser(), a.variants.split(","), Path(a.out).expanduser())
        return
    for dom in a.domains.split(","):
        {"struct": lambda: do_struct(dom), "write": lambda: do_write(dom, a.workers, a.limit),
         "embed": lambda: do_embed(dom, a.batch, a.workers), "score": lambda: do_score(dom)}[a.stage]()


if __name__ == "__main__":
    main()
