#!/usr/bin/env python3
"""Official GPT-5.2 judge, ONLY for the LLM-judged questions (llm_abstention_checker = false-premise,
llm_gotchas_checker = errors-gotchas) of an arm whose harness scoring failed (credits ran out).

Uses the official checker functions of evaluation/qa_eval_metrics.py with the harness's evaluator
settings (gpt-5.2, reasoning_effort medium, max_completion_tokens 4096). Differences from the
harness, all for cost control: no automatic retries (max_retries=0) and the run STOPS at the first
quota / rate-limit error; token usage of every call is logged. The API key is read from ~/e1/.env
and never printed. Rule-scored questions are never sent.
Responses: per_question.jsonl if present, else reader_outputs_cache.jsonl (last valid line).
Writes ~/e1/e2work/official_judge/<arm>_<dom>.jsonl (resumable) and prints token totals.
Usage: python -I e1/analysis/official_judge.py --arm e2_fusion_small [--ids-file F] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.analysis.local_judge import responses  # noqa: E402
from e1.common.io import iter_jsonl  # noqa: E402

E1 = Path("~/e1").expanduser()
LLM = {"llm_abstention_checker", "llm_gotchas_checker"}


class QuotaStop(Exception):
    pass


def api_key() -> str:
    for line in (E1 / ".env").read_text().splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENAI_API_KEY not found in ~/e1/.env")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import evaluation.qa_eval_metrics as Q
    from openai import OpenAI
    key = api_key()
    client = OpenAI(api_key=key, max_retries=0)
    usage_lock = threading.Lock()
    stop = threading.Event()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def call(*, client: object, model, messages, max_completion_tokens, reasoning_effort, temperature, top_p, timeout_seconds):
        if stop.is_set():
            raise QuotaStop("stopped")
        req = {"model": model, "messages": messages, "max_completion_tokens": max_completion_tokens, "timeout": timeout_seconds}
        if reasoning_effort is not None:
            req["reasoning_effort"] = reasoning_effort
        try:
            r = shared.chat.completions.create(**req)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "ratelimit" in msg or "rate limit" in msg or "rate_limit" in msg:
                stop.set()
                raise QuotaStop(str(e)[:200])
            raise
        u = getattr(r, "usage", None)
        with usage_lock:
            usage["calls"] += 1
            usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
        return (r.choices[0].message.content or "").strip()
    shared = client
    Q._call_chat_completion = call
    Q._create_openai_client = lambda **_: client
    keep = {x for x in Path(a.ids_file).expanduser().read_text().split(",") if x} if a.ids_file else None
    out_dir = E1 / "e2work/official_judge"; out_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for dom in ["web", "enterprise"]:
        if stop.is_set():
            break
        items = {it["id"]: it for it in json.loads((E1 / f"runs/b1_small/{dom}_small/runtime_inputs/questions.json").read_text())}
        M = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/meta/{dom}.jsonl")}
        R = responses(a.arm, dom)
        path = out_dir / f"{a.arm}_{dom}.jsonl"
        done = {r["question_id"] for r in iter_jsonl(path) if r.get("error") is None} if path.exists() else set()
        todo = [q for q, m in M.items() if m["eval_function"].split("|")[0] in LLM and q in R and q not in done
                and (keep is None or q in keep)]
        if a.limit is not None:
            todo = todo[: max(0, a.limit - n_done)]

        def one(q):
            r, m = R[q], M[q]
            if stop.is_set():
                return None
            try:
                s = Q.eval_from_spec(m["eval_function"], r["response_raw"], m["answer_gold"], question_item=items[q],
                                     parsed_prediction=r["response_parsed_boxed"], model_response=r["response_raw"],
                                     evaluator_model="gpt-5.2", evaluator_api_key=key, evaluator_reasoning_effort="medium",
                                     evaluator_max_completion_tokens=4096)
                return {"question_id": q, "arm": a.arm, "domain": dom, "question_type": m["question_type"],
                        "score_bool": bool(Q.score_to_bool(s)) and not r["is_unknown"], "error": None}
            except QuotaStop as e:
                return {"question_id": q, "arm": a.arm, "domain": dom, "error": "QUOTA_STOP " + str(e)}
            except Exception as e:  # noqa: BLE001
                return {"question_id": q, "arm": a.arm, "domain": dom, "error": str(e)[:200]}
        with open(path, "a") as fp, ThreadPoolExecutor(a.workers) as ex:
            for row in ex.map(one, todo):
                if row is None:
                    continue
                fp.write(json.dumps(row) + "\n"); fp.flush()
                n_done += row["error"] is None
        print(f"[official_judge] {a.arm} {dom}: sent {len(todo)}; stop={stop.is_set()}", flush=True)
    print(f"[official_judge] usage: calls {usage['calls']}, prompt_tokens {usage['prompt_tokens']}, "
          f"completion_tokens {usage['completion_tokens']}" + ("  ** STOPPED ON QUOTA/RATE ERROR **" if stop.is_set() else ""))
    (out_dir / f"{a.arm}_usage.json").write_text(json.dumps(usage))


if __name__ == "__main__":
    main()
