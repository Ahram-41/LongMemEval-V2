#!/usr/bin/env python3
"""Two reader-side interventions from the error analysis (error-analysis-2026-10-26.md §7.6), tested on
the B1 prompts of all rule-scored, non-abstention questions (web 154 + enterprise 141 = 295).

  base   B1 prompt rows replayed byte-for-byte (a fresh sample with the reasoning kept)
  strip  the same rows with the retrieval-query label lines removed from the memory context
         ("- Source raw query: ...", "- Note query: ...", "- Event query: ..."); nothing else changes
Intervention 1 (wrap-up) is applied inside both variants: when a call ends at the 20k cap with no
answer text, its reasoning is fed back as an assistant prefix that closes the thinking
("<think>\\n" + reasoning + WRAP + "</think>\\n\\n", continue_final_message) and up to 1,024 more tokens
are generated. The un-wrapped answer is kept too, so wrap-up vs no wrap-up is paired on the same sample.

Sampling as the harness: temperature 0.6, top_p 0.95, top_k 20, max_tokens 20000, thinking on.

Modes (cluster, ~/e1 layout):
  run      --variant base|strip  -> ~/e1/runs/iv_<variant>_small/<dom>_small/outputs.jsonl (resumable)
  wrap-old wrap-up on the cap-hit rows of the 2026-10-26 thinking rerun (runs/think_*), quick check
           -> ~/e1/runs/think_<arm>_small/<dom>_small/wrapup.jsonl
  wrap2    forced-format wrap-up on every capped row (iv outputs and the 10-26 think rows): the prefix
           ends with "</think>\n\nThe final answer is \\boxed{" so the model can only write the answer
           (the first wrap-up often restarted the analysis in the answer text and never boxed);
           -> wrap2.jsonl next to each source file
  rep      repeated sampling on a few ids (default: the 6 label-anchor items) under three label
           variants -- base, strip, neutral (each distinct query text replaced by "Q1", "Q2", ... in
           order of first appearance, so blocks retrieved for the same sub-query still share a tag but
           the question's wording is gone) -- --samples N per (id, variant); capped calls get the forced
           wrap-up -> ~/e1/runs/iv_rep_small/outputs.jsonl (resumable) and a per-item summary
  score    rule-score everything -> ~/e1/e2work/reports/iv_scores.jsonl and a printed summary.
           Two scorings: official (the harness: a prediction without \\boxed{} is the whole text, and
           norm_phrase_set_match passes when every gold phrase occurs in it) and strict (a \\boxed{}
           answer must exist).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from evaluation.qa_eval_metrics import eval_from_spec, extract_boxed_answer, is_unknown, score_to_bool  # noqa: E402
from e1.oneoff.rerun_with_thinking import reasoning_of  # noqa: E402

E1 = Path(os.environ.get("E1", "~/e1")).expanduser()
DOMS = ("web", "enterprise")
SRC = E1 / "runs/b1_small"
LABEL_RE = re.compile(r"(?m)^[ \t]*-[ \t]*(?:Source raw query|Note query|Event query)[ \t]*:.*\n?")
WRAP = ("\n\nI have spent my thinking budget. I must stop checking now and give the final answer "
        "based on the thinking above.\n</think>\n\n")
WRAP_TOKENS = 1024
SAMPLING = dict(temperature=0.6, top_p=0.95)
IDS = E1 / "e2work/reports/error_analysis_ids.json"
# error-analysis subgroups (pass 2) whose outcome the interventions predict
LOOP = ["4ed01379", "f1d8a168", "e823a94a", "f61a096f", "e7292dab"]
ANCHOR = ["a45edf3e", "19367bc7", "4df5e6b4", "767e4106", "25b00876", "31f146ba"]


def rule_scored(row) -> bool:
    return not row["is_abstention_problem"] and not row["eval_function"].startswith("llm_")


def strip_labels(messages):
    """Copy of messages with the query-label lines removed; returns (messages, n_removed)."""
    out, n = [], 0
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            parts = []
            for p in c:
                if p.get("type") == "text":
                    t, k = LABEL_RE.subn("", p["text"])
                    n += k
                    p = {**p, "text": t}
                parts.append(p)
            m = {**m, "content": parts}
        out.append(m)
    return out, n


NEUTRAL_RE = re.compile(r"(?m)^([ \t]*-[ \t]*(?:Source raw query|Note query|Event query)[ \t]*:)[ \t]*(.*)$")


def neutral_labels(messages):
    """Copy of messages with each label's query text replaced by a per-prompt id (Q1, Q2, ... by first
    appearance); returns (messages, n_replaced)."""
    ids, out, n = {}, [], 0

    def sub(m):
        nonlocal n
        n += 1
        q = m.group(2).strip()
        ids.setdefault(q, f"Q{len(ids) + 1}")
        return f"{m.group(1)} {ids[q]}"
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = [{**p, "text": NEUTRAL_RE.sub(sub, p["text"])} if p.get("type") == "text" else p for p in c]
            m = {**m, "content": c}
        out.append(m)
    return out, n


def variant_messages(messages, variant):
    if variant == "strip":
        return strip_labels(messages)
    if variant == "neutral":
        return neutral_labels(messages)
    return messages, 0


def load_rows(dom: str, want=None) -> dict:
    rows = {}
    with open(SRC / f"{dom}_small/prompt_rows.jsonl") as fp:
        for line in fp:
            if not line.strip():
                continue
            r = json.loads(line)
            if rule_scored(r) and (want is None or r["question_id"] in want):
                rows[r["question_id"]] = {k: r[k] for k in ("question_id", "question_type", "eval_function",
                                                             "answer_gold", "messages", "memory_context_was_truncated",
                                                             "memory_context_token_count")}
            del r
    return rows


def answer_text(content: str, reasoning: str) -> str:
    """The harness parses the boxed answer from the answer text, or from the reasoning when a capped
    call produced no answer text (response_raw then holds the reasoning)."""
    return content if content.strip() else reasoning


async def wrap_up(client, model, messages, reasoning, sem):
    msgs = messages + [{"role": "assistant", "content": "<think>\n" + reasoning.strip("\n") + WRAP}]
    async with sem:
        r = await client.chat.completions.create(
            model=model, messages=msgs, max_tokens=WRAP_TOKENS, timeout=3600,
            extra_body={"top_k": 20, "continue_final_message": True, "add_generation_prompt": False}, **SAMPLING)
    m = r.choices[0].message
    tail = (reasoning_of(m) or "") + (m.content or "")
    final = tail.split("</think>", 1)[1] if "</think>" in tail else tail
    return {"wrap_text": final.strip(), "wrap_boxed": extract_boxed_answer(final.strip()),
            "wrap_tokens": r.usage.completion_tokens if r.usage else None,
            "wrap_finish": r.choices[0].finish_reason}


async def call_one(client, model, row, messages, args, sem, wrap=True):
    async with sem:
        r = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=args.max_tokens, timeout=args.timeout,
            extra_body={"top_k": 20}, **SAMPLING)
    m = r.choices[0].message
    content = m.content if isinstance(m.content, str) else ""
    reasoning = reasoning_of(m)
    o = {"question_id": row["question_id"], "reasoning": reasoning, "content": content,
         "boxed": extract_boxed_answer(answer_text(content, reasoning).strip()),
         "completion_tokens": r.usage.completion_tokens if r.usage else None,
         "prompt_tokens": r.usage.prompt_tokens if r.usage else None,
         "finish_reason": r.choices[0].finish_reason}
    o["cap_no_answer"] = o["finish_reason"] == "length" and not content.strip()
    if o["cap_no_answer"] and wrap:
        try:
            o.update(await wrap_up(client, model, messages, reasoning, sem))
        except Exception as exc:
            o["wrap_error"] = repr(exc)
    return o


async def run_async(args) -> None:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=os.environ["READER_BASE_URL"], api_key="EMPTY")
    model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
    sem = asyncio.Semaphore(args.concurrency)
    for dom in DOMS:
        dst = E1 / f"runs/iv_{args.variant}_small/{dom}_small"
        dst.mkdir(parents=True, exist_ok=True)
        out = dst / "outputs.jsonl"
        done = set()
        if out.exists():
            for line in open(out):
                try:
                    done.add(json.loads(line)["question_id"])
                except Exception:
                    pass
        rows = load_rows(dom)
        order = sorted(rows)
        random.Random(0).shuffle(order)  # same order in both variants, so partial runs stay paired
        order = [q for q in order if q not in done][: args.limit or None]
        print(f"[iv] {args.variant} {dom}: {len(rows)} rule-scored, {len(done)} done, {len(order)} to run", flush=True)
        jobs = {}
        for q in order:
            msgs, n = (strip_labels(rows[q]["messages"]) if args.variant == "strip" else (rows[q]["messages"], 0))
            jobs[q] = (msgs, n)
        tasks = [call_one(client, model, rows[q], jobs[q][0], args, sem) for q in order]
        with open(out, "a") as fp:
            for fut in asyncio.as_completed(tasks):
                try:
                    o = await fut
                except Exception as exc:  # retried on resume
                    print(f"[iv] request failed: {exc!r}", flush=True)
                    continue
                r = rows[o["question_id"]]
                o.update({"variant": args.variant, "domain": dom, "question_type": r["question_type"],
                          "eval_function": r["eval_function"], "answer_gold": r["answer_gold"],
                          "labels_removed": jobs[o["question_id"]][1],
                          "memory_context_was_truncated": r["memory_context_was_truncated"],
                          "memory_context_token_count": r["memory_context_token_count"]})
                fp.write(json.dumps(o, ensure_ascii=False) + "\n"); fp.flush()
                print(f"[iv] {o['question_id']} tok={o['completion_tokens']} fin={o['finish_reason']} "
                      f"labels-={o['labels_removed']} boxed={o['boxed'][:40]!r} wrap={o.get('wrap_boxed', '')[:40]!r}",
                      flush=True)
    await client.close()


async def wrap_old_async(args) -> None:
    """Wrap-up on the cap-hit rows of the 10-26 thinking rerun (same prompts, same reasoning)."""
    from openai import AsyncOpenAI
    from e1.oneoff.rerun_with_thinking import ARMS, prompt_rows
    client = AsyncOpenAI(base_url=os.environ["READER_BASE_URL"], api_key="EMPTY")
    model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
    sem = asyncio.Semaphore(args.concurrency)
    for arm in ("b1", "fusion"):
        for dom in DOMS:
            d = E1 / f"runs/think_{arm}_small/{dom}_small"
            if not (d / "thinking.jsonl").exists():
                continue
            capped = {}
            for line in open(d / "thinking.jsonl"):
                o = json.loads(line)
                if o["finish_reason"] == "length" and not (o.get("content") or "").strip():
                    capped[o["question_id"]] = o
            rows = prompt_rows(E1 / "runs" / ARMS[arm], dom, set(capped))
            rows.update({q: r for q, r in prompt_rows(E1 / "runs/b1_small", dom, set(capped) - set(rows)).items()})
            print(f"[wrap-old] {arm} {dom}: {len(capped)} capped, {len(rows)} prompts", flush=True)
            res = await asyncio.gather(*[wrap_up(client, model, rows[q]["messages"], capped[q]["reasoning"], sem)
                                         for q in capped if q in rows], return_exceptions=True)
            with open(d / "wrapup.jsonl", "w") as fp:
                for q, w in zip([q for q in capped if q in rows], res):
                    if isinstance(w, Exception):
                        w = {"wrap_error": repr(w)}
                    o = capped[q]
                    ok = None
                    if "wrap_boxed" in w:
                        ok = (not is_unknown(w["wrap_boxed"])) and score_to_bool(
                            eval_from_spec(o["eval_function"], w["wrap_boxed"], o["answer_gold"]))
                    rec = {"question_id": q, "arm": arm, "domain": dom, "answer_gold": o["answer_gold"],
                           "eval_function": o["eval_function"], "wrap_correct": ok, **w}
                    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"[wrap-old] {arm} {q} gold={o['answer_gold']!r} wrap={w.get('wrap_boxed')!r} ok={ok}", flush=True)
    await client.close()


WRAP2 = WRAP + "The final answer is \\boxed{"


async def wrap_forced(client, model, messages, reasoning, sem):
    msgs = messages + [{"role": "assistant", "content": "<think>\n" + reasoning.strip("\n") + WRAP2}]
    async with sem:
        r = await client.chat.completions.create(
            model=model, messages=msgs, max_tokens=64, timeout=3600,
            extra_body={"top_k": 20, "continue_final_message": True, "add_generation_prompt": False}, **SAMPLING)
    m = r.choices[0].message
    tail = (m.content or "") + (reasoning_of(m) or "")
    ans = extract_boxed_answer("\\boxed{" + tail)
    return {"wrap2_boxed": ans if "}" in tail else ans.split("\n", 1)[0], "wrap2_raw": tail[:300]}


async def wrap2_async(args) -> None:
    from openai import AsyncOpenAI
    from e1.oneoff.rerun_with_thinking import ARMS, prompt_rows
    client = AsyncOpenAI(base_url=os.environ["READER_BASE_URL"], api_key="EMPTY")
    model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
    sem = asyncio.Semaphore(args.concurrency)
    srcs = [(E1 / f"runs/iv_{v}_small/{d}_small/outputs.jsonl", v) for v in ("base", "strip") for d in DOMS]
    srcs += [(E1 / f"runs/think_{a}_small/{d}_small/thinking.jsonl", a) for a in ("b1", "fusion") for d in DOMS]
    for path, tag in srcs:
        if not path.exists():
            continue
        dom = path.parent.name.replace("_small", "")
        capped = {}
        for line in open(path):
            o = json.loads(line)
            if o["finish_reason"] == "length" and not (o.get("content") or "").strip():
                capped[o["question_id"]] = o
        out = path.parent / "wrap2.jsonl"
        done = {json.loads(l)["question_id"] for l in open(out)} if out.exists() else set()
        todo = [q for q in capped if q not in done]
        if tag in ("base", "strip"):
            rows = load_rows(dom, set(todo))
            if tag == "strip":
                rows = {q: {**r, "messages": strip_labels(r["messages"])[0]} for q, r in rows.items()}
        else:
            rows = prompt_rows(E1 / "runs" / ARMS[tag], dom, set(todo))
            rows.update(prompt_rows(E1 / "runs/b1_small", dom, set(todo) - set(rows)))
        print(f"[wrap2] {path}: {len(capped)} capped, {len(todo)} to do", flush=True)
        res = await asyncio.gather(*[wrap_forced(client, model, rows[q]["messages"], capped[q]["reasoning"], sem)
                                     for q in todo], return_exceptions=True)
        with open(out, "a") as fp:
            for q, w in zip(todo, res):
                if isinstance(w, Exception):
                    print(f"[wrap2] {q} failed {w!r}", flush=True)
                    continue
                o = capped[q]
                ok = correct(w["wrap2_boxed"], o)
                fp.write(json.dumps({"question_id": q, "source": tag, "answer_gold": o["answer_gold"],
                                     "wrap2_correct": ok, **w}, ensure_ascii=False) + "\n")
                print(f"[wrap2] {tag} {q} gold={o['answer_gold']!r} wrap2={w['wrap2_boxed']!r} ok={ok}", flush=True)
    await client.close()


async def rep_async(args) -> None:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=os.environ["READER_BASE_URL"], api_key="EMPTY")
    model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
    sem = asyncio.Semaphore(args.concurrency)
    ids = args.ids.split(",") if args.ids else ANCHOR
    rows = {}
    for dom in DOMS:
        for q, r in load_rows(dom, set(ids)).items():
            rows[q] = {**r, "domain": dom}
    out = E1 / "runs/iv_rep_small/outputs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            o = json.loads(line)
            done.add((o["variant"], o["question_id"], o["sample"]))
    variants = args.variants.split(",")
    todo = [(v, q, s) for s in range(args.samples) for q in ids for v in variants if (v, q, s) not in done]
    print(f"[rep] {len(ids)} ids x {variants} x {args.samples}: {len(done)} done, {len(todo)} to run", flush=True)

    async def one(v, q, s):
        msgs, n = variant_messages(rows[q]["messages"], v)
        o = await call_one(client, model, rows[q], msgs, args, sem, wrap=False)
        if o["cap_no_answer"]:
            o.update(await wrap_forced(client, model, msgs, o["reasoning"], sem))
        o.update({"variant": v, "sample": s, "labels_changed": n})
        return o
    tasks = [one(*t) for t in todo]
    with open(out, "a") as fp:
        for fut in asyncio.as_completed(tasks):
            try:
                o = await fut
            except Exception as exc:
                print(f"[rep] request failed: {exc!r}", flush=True)
                continue
            r = rows[o["question_id"]]
            o.update({k: r[k] for k in ("domain", "question_type", "eval_function", "answer_gold")})
            o["official"] = correct(o["boxed"], o)
            o["strict"] = correct_strict(o["content"], o)
            o["forced"] = bool(correct(o["wrap2_boxed"], o)) if o["cap_no_answer"] else o["strict"]
            fp.write(json.dumps(o, ensure_ascii=False) + "\n"); fp.flush()
            print(f"[rep] {o['variant']:7} {o['question_id']} s{o['sample']} tok={o['completion_tokens']} "
                  f"cap={o['cap_no_answer']} forced={o['forced']} ans={(o.get('wrap2_boxed') or o['boxed'])[:40]!r}",
                  flush=True)
    await client.close()
    rep_summary(variants)


def rep_summary(variants=("base", "strip", "neutral")) -> None:
    res = defaultdict(lambda: defaultdict(list))
    for line in open(E1 / "runs/iv_rep_small/outputs.jsonl"):
        o = json.loads(line)
        res[o["question_id"]][o["variant"]].append(o)
    tot = Counter()
    for q in sorted(res):
        cells = []
        for v in variants:
            R = res[q][v]
            k = sum(o["forced"] for o in R); c = sum(o["cap_no_answer"] for o in R)
            tot[v, "k"] += k; tot[v, "n"] += len(R); tot[v, "cap"] += c
            cells.append(f"{v} {k}/{len(R)} (cap {c})")
        print(f"  {q} gold={res[q][variants[0]][0]['answer_gold'][:30]!r:32} " + " | ".join(cells))
    print("  total " + " | ".join(f"{v} {tot[v, 'k']}/{tot[v, 'n']} (cap {tot[v, 'cap']})" for v in variants))


# ----------------------------------------------------------------------------- score
def correct(boxed: str, row) -> bool:
    return (not is_unknown(boxed)) and score_to_bool(eval_from_spec(row["eval_function"], boxed, row["answer_gold"]))


def correct_strict(text: str, row) -> bool:
    """Only an explicit \\boxed{} answer counts."""
    return "\\boxed{" in text and correct(extract_boxed_answer(text.strip()), row)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial p for b vs c discordant pairs."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def score(args) -> None:
    orig = {}
    for dom in DOMS:
        with open(SRC / f"{dom}_small/per_question.jsonl") as fp:
            for line in fp:
                r = json.loads(line)
                if rule_scored(r):
                    orig[r["question_id"]] = bool(r["score_bool"])
                del r
    res = {}
    w2 = {}
    for v in ("base", "strip"):
        for dom in DOMS:
            p2 = E1 / f"runs/iv_{v}_small/{dom}_small/wrap2.jsonl"
            if p2.exists():
                w2[(v, dom)] = {json.loads(l)["question_id"]: json.loads(l) for l in open(p2)}
    for v in ("base", "strip"):
        for dom in DOMS:
            p = E1 / f"runs/iv_{v}_small/{dom}_small/outputs.jsonl"
            if not p.exists():
                continue
            for line in open(p):
                o = json.loads(line)
                q = o["question_id"]
                rec = {"question_id": q, "variant": v, "domain": dom, "question_type": o["question_type"],
                       "plain": correct(o["boxed"], o), "cap": o["cap_no_answer"],
                       "completion_tokens": o["completion_tokens"], "labels_removed": o["labels_removed"]}
                rec["wrapped"] = correct(o["wrap_boxed"], o) if o["cap_no_answer"] and "wrap_boxed" in o else rec["plain"]
                rec["plain_strict"] = correct_strict(o["content"], o)
                rec["wrapped_strict"] = (correct_strict(o.get("wrap_text", ""), o) if o["cap_no_answer"]
                                         else rec["plain_strict"])
                rec["wrap2"] = w2.get((v, dom), {}).get(q, {}).get("wrap2_correct") if o["cap_no_answer"] else None
                rec["forced"] = rec["wrap2"] if rec["wrap2"] is not None else rec["plain_strict"]
                res[(v, q)] = rec
    out = E1 / "e2work/reports/iv_scores.jsonl"
    with open(out, "w") as fp:
        for rec in res.values():
            fp.write(json.dumps({**rec, "b1_original": orig.get(rec["question_id"])}) + "\n")
    both = sorted(q for (v, q) in res if v == "base" and ("strip", q) in res)
    print(f"[score] base {sum(v == 'base' for v, _ in res)}, strip {sum(v == 'strip' for v, _ in res)}, paired {len(both)}  -> {out}")
    for v in ("base", "strip"):
        R = [r for (vv, _), r in res.items() if vv == v]
        if not R:
            continue
        n = len(R); cap = sum(r["cap"] for r in R)
        gain = sum(r["wrapped"] and not r["plain"] for r in R); loss = sum(r["plain"] and not r["wrapped"] for r in R)
        print(f"  {v:5} n={n} cap {cap} | official: plain {sum(r['plain'] for r in R)} ({sum(r['plain'] for r in R)/n:.3f}) "
              f"wrapped {sum(r['wrapped'] for r in R)} (gain {gain} loss {loss}) | strict: plain {sum(r['plain_strict'] for r in R)} "
              f"wrapped {sum(r['wrapped_strict'] for r in R)} forced {sum(bool(r['forced']) for r in R)} "
              f"(forced on cap: {sum(bool(r['wrap2']) for r in R if r['cap'])}/{sum(r['wrap2'] is not None for r in R)})")
        bo = [r for r in R if r["question_id"] in orig]
        b = sum(r["plain"] and not orig[r["question_id"]] for r in bo); c = sum(orig[r["question_id"]] and not r["plain"] for r in bo)
        print(f"        vs B1 original (plain): +{b} -{c}  p={mcnemar(b, c):.3f}")
    for key in ("plain", "wrapped", "plain_strict", "forced"):
        b = sum(res[("strip", q)][key] and not res[("base", q)][key] for q in both)
        c = sum(res[("base", q)][key] and not res[("strip", q)][key] for q in both)
        print(f"  strip vs base ({key}) on {len(both)} paired: strip fixes {b}, breaks {c}, p={mcnemar(b, c):.3f}")
        by = defaultdict(Counter)
        for q in both:
            k = (res[("base", q)]["domain"], res[("base", q)]["question_type"])
            by[k]["n"] += 1; by[k]["base"] += res[("base", q)][key]; by[k]["strip"] += res[("strip", q)][key]
        for k, c_ in sorted(by.items()):
            print(f"      {k[0]:10} {k[1]:20} n={c_['n']:3} base {c_['base']:3} strip {c_['strip']:3}")
    for name, ids in (("loop", LOOP), ("anchor", ANCHOR)):
        for q in ids:
            cells = [f"{v}:{'x' if (v, q) not in res else ('cap ' if res[(v, q)]['cap'] else '') + ('P' if res[(v, q)]['plain'] else '-') + '/' + ('W' if res[(v, q)]['wrapped'] else '-')}"
                     for v in ("base", "strip")]
            print(f"  {name:6} {q} B1orig={orig.get(q)} " + " ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "wrap-old", "wrap2", "score", "rep", "rep-summary"])
    ap.add_argument("--ids", default=None)
    ap.add_argument("--variants", default="base,strip,neutral")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--variant", choices=["base", "strip"], default="base")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "run":
        asyncio.run(run_async(a))
    elif a.mode == "wrap-old":
        asyncio.run(wrap_old_async(a))
    elif a.mode == "wrap2":
        asyncio.run(wrap2_async(a))
    elif a.mode == "rep":
        asyncio.run(rep_async(a))
    elif a.mode == "rep-summary":
        rep_summary(tuple(a.variants.split(",")))
    else:
        score(a)


if __name__ == "__main__":
    main()
