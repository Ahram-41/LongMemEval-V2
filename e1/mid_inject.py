#!/usr/bin/env python3
"""Mid-thinking injection: let the reader think; if it is still thinking after K tokens,
insert a short passage into its own reasoning and let it continue (total budget unchanged).

Arms
  mi : inject the premise-check verdict (same content as the P-any note), first person
  mn : inject a verdict-free nudge to stop re-checking and decide (control)
Scope: the questions that received a P-any note (others would get no injection and are identical
to B1). Prompts are B1's rows (fixed system prompt, same retrieved context).

Stage 1: chat completion with max_tokens=K. If it ends by length while still thinking (no final
content), Stage 2 continues the assistant turn: messages + assistant prefix
"<think>\n" + partial_reasoning + injection, with continue_final_message=True,
add_generation_prompt=False, max_tokens = BUDGET - K.
Writes <dst>/<dom>_small/reader_outputs_cache.jsonl (harness format, resumable) plus
mid_inject_log.jsonl, and copies B1 prompt rows for those questions so run_e1 only scores.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "e1"))
from evaluation.qa_eval_metrics import extract_boxed_answer, is_unknown  # noqa: E402
from derive_arm import note_from_verdict  # noqa: E402

MI_TEMPLATE = (
    "\n\nWait. Before going further, let me use the automatic premise check of the retrieved memory. "
    "It reports that the memory appears to contradict an assumption in the question:\n"
    "- Assumption: {assumption}\n- What the memory shows: {evidence}\n"
    "If that holds, the question's premise is wrong, and instead of answering it I should explain inside "
    "\\boxed{{}} why the question is flawed. Let me verify this against the evidence quickly and then decide.\n"
)
MN_TEXT = (
    "\n\nWait. I have been going back and forth for a long time. Let me stop re-checking and decide now "
    "from the evidence I already have: if the thing the question takes for granted is not supported by the "
    "memory, the premise is probably wrong and I should explain that inside \\boxed{}; otherwise I give my "
    "best answer.\n"
)


def reasoning_of(msg) -> str:
    for k in ("reasoning_content", "reasoning"):
        v = getattr(msg, k, None)
        if isinstance(v, str) and v:
            return v
    extra = getattr(msg, "model_extra", None) or {}
    for k in ("reasoning_content", "reasoning"):
        if isinstance(extra.get(k), str) and extra[k]:
            return extra[k]
    return ""


def injection_text(arm: str, verdict: dict, question: str) -> str:
    if arm == "mn":
        return MN_TEXT
    note = note_from_verdict(verdict, question, "any") or ""
    a = e = "(unspecified)"
    for line in note.splitlines():
        if line.startswith("- Assumption:"):
            a = line.split(":", 1)[1].strip()
        if line.startswith("- What the memory shows:"):
            e = line.split(":", 1)[1].strip()
    return MI_TEMPLATE.format(assumption=a, evidence=e)


async def run_one(client, model, row, inj, args, sem):
    sampling = dict(temperature=0.6, top_p=0.95)
    async with sem:
        r1 = await client.chat.completions.create(
            model=model, messages=row["messages"], max_tokens=args.k, timeout=3600,
            extra_body={"top_k": 20}, **sampling)
        m1 = r1.choices[0].message
        reas1, cont1 = reasoning_of(m1), (m1.content or "")
        tok1 = r1.usage.completion_tokens if r1.usage else 0
        still_thinking = r1.choices[0].finish_reason == "length" and not cont1.strip()
        log = {"question_id": row["question_id"], "stage1_tokens": tok1, "injected": False}
        answer_cut = r1.choices[0].finish_reason == "length" and bool(cont1.strip())
        if answer_cut:
            # thinking finished before K but the answer was cut: continue it, no injection
            prefix = "<think>\n" + reas1.strip("\n") + "\n</think>\n\n" + cont1
            msgs = row["messages"] + [{"role": "assistant", "content": prefix}]
            r2 = await client.chat.completions.create(
                model=model, messages=msgs, max_tokens=args.budget - args.k, timeout=3600,
                extra_body={"top_k": 20, "continue_final_message": True, "add_generation_prompt": False},
                **sampling)
            m2 = r2.choices[0].message
            final_raw = cont1 + (m2.content or "") + reasoning_of(m2)
            total = tok1 + (r2.usage.completion_tokens if r2.usage else 0)
            log.update({"answer_continued": True, "stage2_tokens": total - tok1})
        elif not still_thinking:
            text = cont1 if cont1.strip() else reas1
            final_raw, total = text, tok1
        else:
            prefix = "<think>\n" + reas1 + inj
            msgs = row["messages"] + [{"role": "assistant", "content": prefix}]
            r2 = await client.chat.completions.create(
                model=model, messages=msgs, max_tokens=args.budget - args.k, timeout=3600,
                extra_body={"top_k": 20, "continue_final_message": True, "add_generation_prompt": False},
                **sampling)
            m2 = r2.choices[0].message
            cont2 = (m2.content or "")
            reas2 = reasoning_of(m2)
            tail = reas2 + cont2
            # final answer = text after the closing think tag if present
            final_raw = tail.split("</think>", 1)[1] if "</think>" in tail else (cont2 if cont2.strip() else tail)
            total = tok1 + (r2.usage.completion_tokens if r2.usage else 0)
            log.update({"injected": True, "stage2_tokens": total - tok1,
                        "stage2_finish": r2.choices[0].finish_reason,
                        "stage2_has_close_tag": "</think>" in tail, "stage2_content_len": len(cont2),
                        "stage2_reasoning_len": len(reas2), "tail_head": tail[:300]})
        parsed = extract_boxed_answer(final_raw.strip())
        out = {"response_raw": final_raw.strip(), "response_parsed_boxed": parsed,
               "is_unknown": is_unknown(parsed),
               "usage": {"prompt_tokens": 0, "completion_tokens": total, "total_tokens": total}}
        return row["question_id"], out, log


async def main_async(args):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=os.environ["READER_BASE_URL"], api_key="EMPTY")
    model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
    sem = asyncio.Semaphore(args.concurrency)
    runs = Path(args.runs).expanduser()
    for dom in ["web_small", "enterprise_small"]:
        ids = [x for x in (runs / "pa_small" / dom / "noted_ids.txt").read_text().split(",") if x]
        if args.limit:
            ids = ids[: args.limit]
        verdicts = {json.loads(l)["question_id"]: json.loads(l) for l in open(runs / "p_small" / dom / "premise_check.jsonl")}
        dst = runs / f"{args.arm}_small" / dom
        dst.mkdir(parents=True, exist_ok=True)
        rows_out = dst / "prompt_rows.jsonl"
        cache = dst / "reader_outputs_cache.jsonl"
        done = set()
        if cache.exists():
            for l in open(cache):
                try:
                    done.add(json.loads(l)["question_id"])
                except Exception:
                    pass
        want = set(ids)
        rows = []
        with open(runs / "b1_small" / dom / "prompt_rows.jsonl") as f:
            for l in f:
                if not l.strip():
                    continue
                r = json.loads(l)
                if r["question_id"] in want:
                    rows.append(r)
        if not rows_out.exists():
            with open(rows_out, "w") as fp:
                for r in rows:
                    fp.write(json.dumps(r) + "\n")
        (dst / "noted_ids.txt").write_text(",".join(r["question_id"] for r in rows))
        todo = [r for r in rows if r["question_id"] not in done]
        print(f"[mid_inject] {args.arm} {dom}: {len(rows)} questions, {len(done)} cached, {len(todo)} to run", flush=True)
        tasks = [run_one(client, model, r, injection_text(args.arm, verdicts.get(r["question_id"], {}), r["question_text"]), args, sem)
                 for r in todo]
        with open(cache, "a") as cfp, open(dst / "mid_inject_log.jsonl", "a") as lfp:
            for fut in asyncio.as_completed(tasks):
                qid, out, log = await fut
                cfp.write(json.dumps({"question_id": qid, "output": out}) + "\n"); cfp.flush()
                lfp.write(json.dumps(log) + "\n"); lfp.flush()
                print(f"[mid_inject] {qid} injected={log['injected']} tokens={out['usage']['completion_tokens']} boxed={out['response_parsed_boxed'][:60]!r}", flush=True)
    await client.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["mi", "mn"], required=True)
    ap.add_argument("--runs", default="~/e1/runs")
    ap.add_argument("--k", type=int, default=12000)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
