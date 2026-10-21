#!/usr/bin/env python3
"""LOCAL judge fallback (GPT-5.2 credits exhausted 2026-10-22 05:05).

本地评分，与官方 GPT-5.2 评分不可比，只用于配对比较 (local scores are NOT comparable to the official
GPT-5.2 judge; they are only for paired comparisons between arms under the same local judge).

For the LLM-judged questions (llm_abstention_checker = false-premise, llm_gotchas_checker =
errors-gotchas) it calls the OFFICIAL checker functions of evaluation/qa_eval_metrics.py (official
judge prompts and parsing) with the evaluator pointed at the local Qwen3.5-9B server, temperature 0,
thinking disabled (chat_template_kwargs.enable_thinking=False is added to the official request).
Responses: B1 from per_question.jsonl; other arms from reader_outputs_cache.jsonl (last valid line).
Writes ~/e1/e2work/local_judge/<arm>_<dom>.jsonl (never touches run folders) and, for b1_small,
the agreement / Cohen's kappa with B1's GPT-5.2 labels.
Usage (GPU node, reader server up): python e1/analysis/local_judge.py --arms b1_small,e2_fusion_small
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402

E1 = Path("~/e1").expanduser()
LLM = {"llm_abstention_checker", "llm_gotchas_checker"}


def patch_no_thinking():
    import evaluation.qa_eval_metrics as Q
    orig = Q._call_chat_completion

    def call(*, client, model, messages, max_completion_tokens, reasoning_effort, temperature, top_p, timeout_seconds):
        r = client.chat.completions.create(model=model, messages=messages, max_completion_tokens=max_completion_tokens,
                                           temperature=temperature, timeout=timeout_seconds,
                                           extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return (r.choices[0].message.content or "").strip()
    Q._call_chat_completion = call
    return Q, orig


def responses(arm: str, dom: str) -> dict:
    p = E1 / f"runs/{arm}/{dom}_small/per_question.jsonl"
    out = {}
    if p.exists() and arm == "b1_small":
        for r in iter_jsonl(p):
            out[r["question_id"]] = {"response_raw": r["response_raw"], "response_parsed_boxed": r["response_parsed_boxed"],
                                     "is_unknown": r["is_unknown"], "official": bool(r["score_bool"])}
            del r
        return out
    with open(E1 / f"runs/{arm}/{dom}_small/reader_outputs_cache.jsonl", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            o = c.get("output")
            if isinstance(o, dict):
                out[c["question_id"]] = {"response_raw": o["response_raw"], "response_parsed_boxed": o["response_parsed_boxed"],
                                         "is_unknown": o["is_unknown"], "official": None}
    return out


def kappa(pairs):
    n = len(pairs)
    if not n:
        return float("nan"), float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="b1_small,e2_fusion_small")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--ids-file", default=None)
    a = ap.parse_args()
    Q, _ = patch_no_thinking()
    base = os.environ.get("READER_BASE_URL", "http://localhost:8023/v1")
    model = os.environ.get("READER_MODEL", "Qwen/Qwen3.5-9B")
    keep = {x for x in Path(a.ids_file).expanduser().read_text().split(",") if x} if a.ids_file else None
    out_dir = E1 / "e2work/local_judge"; out_dir.mkdir(parents=True, exist_ok=True)
    for arm in a.arms.split(","):
        for dom in ["web", "enterprise"]:
            items = {it["id"] if "id" in it else it["question_id"]: it
                     for it in json.loads((E1 / f"runs/b1_small/{dom}_small/runtime_inputs/questions.json").read_text())}
            M = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/meta/{dom}.jsonl")}
            R = responses(arm, dom)
            path = out_dir / f"{arm}_{dom}.jsonl"
            done = {r["question_id"] for r in iter_jsonl(path)} if path.exists() else set()
            todo = [q for q, m in M.items() if m["eval_function"].split("|")[0] in LLM and q in R and q not in done
                    and (keep is None or q in keep)]

            def one(q):
                r, m = R[q], M[q]
                try:
                    s = Q.eval_from_spec(m["eval_function"], r["response_raw"], m["answer_gold"],
                                         question_item=items[q], parsed_prediction=r["response_parsed_boxed"],
                                         model_response=r["response_raw"], evaluator_model=model, evaluator_base_url=base,
                                         evaluator_api_key="EMPTY", evaluator_temperature=0.0,
                                         evaluator_max_completion_tokens=2048, evaluator_timeout_seconds=600)
                    ok = Q.score_to_bool(s) and not r["is_unknown"]
                    err = None
                except Exception as e:  # noqa: BLE001
                    ok, err = None, str(e)[:200]
                return {"question_id": q, "arm": arm, "domain": dom, "question_type": m["question_type"],
                        "local_judge": ok, "official_gpt52": r["official"], "error": err,
                        "label": "本地评分，与官方 GPT-5.2 评分不可比，只用于配对比较"}
            with open(path, "a") as fp, ThreadPoolExecutor(a.workers) as ex:
                for row in ex.map(one, todo):
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n"); fp.flush()
            print(f"[local_judge] {arm} {dom}: {len(todo)} judged -> {path}", flush=True)
    # calibration on B1
    pairs = []
    for dom in ["web", "enterprise"]:
        p = out_dir / f"b1_small_{dom}.jsonl"
        if p.exists():
            pairs += [(bool(r["local_judge"]), bool(r["official_gpt52"])) for r in iter_jsonl(p)
                      if r["local_judge"] is not None and r["official_gpt52"] is not None]
    po, k = kappa(pairs)
    print(f"[local_judge] calibration on B1: n={len(pairs)} agreement={po:.3f} Cohen kappa={k:.3f} "
          f"(local positive {sum(a for a, _ in pairs)}, GPT-5.2 positive {sum(b for _, b in pairs)})")


if __name__ == "__main__":
    main()
