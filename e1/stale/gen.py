#!/usr/bin/env python3
"""STALE reader pass: the five counterfactual contexts of mem-test v2, answered through an OpenAI-compatible
reader (READER_BASE_URL, default Qwen/Qwen3.5-9B from serve_vllm.sh).

Context construction is verbatim from stale_gen_v2.py (same order of evaluation, so the shared `used` set
gives the same distractors). Prompt modes:
  v2       : mem-test v2 system prompt (hints at recency/change) + "Excerpts:" lines "[date] role: text" in
             retrieval rank order (--order chrono sorts them by unit index = time).
  official : STALE run_target_model.build_prompts system/user prompts (no recency hint); the retrieved turns are
             rendered with its format_haystack layout (=== Session i [Time: t] ===, User:/Assistant:), sessions
             and turns in chronological order.
Reader: temperature 0, thinking off unless --thinking; max tokens 160/400/400 (v2) unless --max-tokens N.
Resumable: every reader call is cached in RUNS/<arm>/reader_cache.jsonl keyed by the exact request.
Output RUNS/<arm>/gen.jsonl rows have the stale_gen_v2.jsonl schema (+ type, prompt params).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.manifest import write_manifest  # noqa: E402
from e1.stale.common import CONDS, DIMS, K, RET, RUNS, client, fmt, iter_jsonl, load_scenarios  # noqa: E402

MAXTOK_V2 = {"dim1_query": 160, "dim2_query": 400, "dim3_query": 400}
SYS_V2 = ("You are a personal assistant with access to retrieved excerpts of the user's past conversations. Excerpts are timestamped. "
          "Use the most recent applicable information about the user's current situation; note when something has changed. Be concise.")
# STALE/Evaluation/run_target_model.py build_prompts (verbatim)
SYS_OFF_QA = ("You are a helpful assistant. Review the following conversation history "
              "with the user, then accurately answer the question.")
SYS_OFF_TASK = ("You are a helpful assistant. Review the following conversation history "
                "with the user, then respond to the user's latest query directly.")


def contexts(sc: dict, order: list[int]) -> tuple[dict, dict]:
    """The five conditions for one probe (verbatim logic of stale_gen_v2.py)."""
    units, j_old, j_new = sc["units"], sc["j_old"], sc["j_new"]
    is_old = lambda j: units[j]["sid"] == sc["old_sid"]  # noqa: E731
    is_new = lambda j: units[j]["sid"] == sc["new_sid"]  # noqa: E731
    top = order[:K]
    topset = set(top)
    used = set(top)

    def distractor(j):  # length-matched turn from a neutral session, not yet used in this probe
        L = len(units[j]["text"])
        cands = [i for i in range(len(units)) if i not in used and not is_old(i) and not is_new(i)]
        i = min(cands, key=lambda i: abs(len(units[i]["text"]) - L))
        used.add(i)
        return i

    def refill(base, excl):  # pad with next-ranked turns not violating excl, keep K
        out = list(base)
        for j in order:
            if len(out) >= K:
                break
            if j not in out and not excl(j):
                out.append(j)
        assert len(out) >= K, "retrieval order too short for refill; raise TOPN"
        return out[:K]
    conds = {
        "C0": top,
        "M_stale": [distractor(j) if is_old(j) else j for j in top],
        "S_rm_current": refill([j for j in top if not is_new(j)], is_new),
        "O_both": ([j_old, j_new] + [j for j in top if j not in (j_old, j_new)])[:K],
        "O_new_only": refill([j_new] + [j for j in top if not is_old(j) and j != j_new], is_old),
    }
    meta = {"old_in_top": j_old in topset, "new_in_top": j_new in topset,
            "n_old_sess": sum(is_old(j) for j in top), "n_new_sess": sum(is_new(j) for j in top)}
    return conds, meta


def messages(sc: dict, dim: str, idxs: list[int], mode: str, order: str) -> list[dict]:
    units, query = sc["units"], sc["probing_queries"][dim]
    if mode == "v2":
        seq = sorted(idxs) if order == "chrono" else idxs
        ctx = "\n".join(fmt(units[j]) for j in seq)
        return [{"role": "system", "content": SYS_V2}, {"role": "user", "content": f"Excerpts:\n{ctx}\n\nUser message: {query}"}]
    hist, cur = "", None
    for j in sorted(idxs):
        u = units[j]
        if u["sid"] != cur:
            cur = u["sid"]
            hist += f"\n=== Session {cur + 1} [Time: {u['date']}] ===\n"
        hist += f"{'User' if u['role'] == 'user' else 'Assistant'}: {u['text']}\n"
    if dim == "dim3_query":
        return [{"role": "system", "content": SYS_OFF_TASK}, {"role": "user", "content": f"[Conversation History]\n{hist}\n\n[Latest Query]\n{query}"}]
    return [{"role": "system", "content": SYS_OFF_QA}, {"role": "user", "content": f"[Conversation History]\n{hist}\n\n[Question]\n{query}"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="t1_69")
    ap.add_argument("--retriever", default="bge")
    ap.add_argument("--arm", required=True, help="output folder name under RUNS")
    ap.add_argument("--prompt", default="v2", choices=["v2", "official"])
    ap.add_argument("--order", default="rank", choices=["rank", "chrono"])
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = v2 budgets 160/400/400")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
    cl = client(os.environ["READER_BASE_URL"])
    dst = RUNS / a.arm
    dst.mkdir(parents=True, exist_ok=True)
    conds_keep = a.conds.split(",")
    scen = load_scenarios(a.set)[: a.limit or None]
    orders = {r["uid"]: r["order"] for r in iter_jsonl(RET / a.retriever / f"{a.set}.jsonl")}

    cache_p = dst / "reader_cache.jsonl"
    cache = {}
    if cache_p.exists():
        for r in iter_jsonl(cache_p):
            cache[r["key"]] = r
    lock = threading.Lock()
    fc = open(cache_p, "a")

    plan, todo = [], {}
    for sc in scen:
        for dim in DIMS:
            conds, meta = contexts(sc, orders[sc["uid"]][dim])
            mt = a.max_tokens or MAXTOK_V2[dim]
            if a.thinking and not a.max_tokens:
                mt += 8000
            keys = {}
            for c in conds_keep:
                msgs = messages(sc, dim, conds[c], a.prompt, a.order)
                req = {"model": model, "messages": msgs, "max_tokens": mt, "thinking": a.thinking}
                k = hashlib.md5(json.dumps(req, sort_keys=True).encode()).hexdigest()
                keys[c] = k
                if k not in cache:
                    todo[k] = req
            plan.append((sc, dim, conds, meta, keys))
    print(f"[stale gen] {a.arm}: {len(scen)} scenarios, {len(plan) * len(conds_keep)} answers, {len(todo)} reader calls to make", flush=True)

    def call(item):
        k, req = item
        t = time.time()
        r = cl.chat.completions.create(model=req["model"], messages=req["messages"], max_tokens=req["max_tokens"], temperature=0,
                                       extra_body={"chat_template_kwargs": {"enable_thinking": req["thinking"]}})
        m = r.choices[0].message
        row = {"key": k, "answer": (m.content or "").strip(), "reasoning": getattr(m, "reasoning_content", None) or "",
               "finish_reason": r.choices[0].finish_reason, "completion_tokens": r.usage.completion_tokens,
               "prompt_tokens": r.usage.prompt_tokens, "seconds": round(time.time() - t, 2)}
        with lock:
            cache[k] = row
            fc.write(json.dumps(row) + "\n")
            fc.flush()
        return k
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(call, todo.items()), 1):
            if i % 100 == 0 or i == len(todo):
                print(f"[stale gen] {i}/{len(todo)} calls {time.time() - t0:.0f}s", flush=True)

    with open(dst / "gen.jsonl", "w") as f:
        recs = {}
        for sc, dim, conds, meta, keys in plan:
            rec = recs.setdefault(sc["uid"], {k: sc[k] for k in ("uid", "type", "M_old", "M_new", "explanation", "j_old", "j_new",
                                                                   "match_old", "match_new", "old_sid", "new_sid")} | {"probes": {}})
            rec["probes"][dim] = {"query": sc["probing_queries"][dim], **meta, "ctx": {c: conds[c] for c in conds_keep},
                                  "answers": {c: cache[keys[c]]["answer"] for c in conds_keep},
                                  "finish": {c: cache[keys[c]]["finish_reason"] for c in conds_keep}}
        for rec in recs.values():
            f.write(json.dumps(rec) + "\n")
    cut = sum(p["finish"][c] == "length" for r in recs.values() for p in r["probes"].values() for c in conds_keep)
    print(f"[stale gen] wrote {dst / 'gen.jsonl'} ({len(recs)} scenarios); answers cut at max_tokens: {cut}", flush=True)
    write_manifest(dst, inputs={"prep": a.set, "retrieval": RET / a.retriever / f"{a.set}.jsonl"},
                   params={k: v for k, v in vars(a).items()} | {"reader_model": model}, name="manifest_gen.json")


if __name__ == "__main__":
    main()
