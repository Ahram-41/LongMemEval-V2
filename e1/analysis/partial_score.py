#!/usr/bin/env python3
"""Score cached reader outputs with the official rule-based scorers only (no LLM judge).

Questions graded by llm_abstention_checker / llm_gotchas_checker are skipped. Uses
evaluation/qa_eval_metrics.py directly (no torch/transformers), so it runs on the login node.
Writes <run_dir>/partial_rule_scores.jsonl (one row per scored question) and prints a table.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.qa_eval_metrics import eval_from_spec, eval_name, score_to_bool  # noqa: E402

CAT = {"static-environment": "static", "dynamic-environment": "dynamic", "procedure": "workflow",
       "errors-gotchas": "gotchas"}
LLM = {"llm_abstention_checker", "llm_gotchas_checker"}

def main(run_root: str) -> None:
    root = Path(run_root)
    tot = defaultdict(lambda: [0, 0]); capped = defaultdict(lambda: [0, 0]); skipped = 0
    for dom_dir in sorted(root.glob("*_small")):
        prompts = {}
        for line in open(dom_dir / "prompt_rows.jsonl"):
            r = json.loads(line); prompts[r["question_id"]] = r
        out_rows = []
        for line in open(dom_dir / "reader_outputs_cache.jsonl"):
            c = json.loads(line); qid = c["question_id"]; o = c["output"]; p = prompts[qid]
            if eval_name(p["eval_function"]) in LLM:
                skipped += 1; continue
            ok = score_to_bool(eval_from_spec(p["eval_function"], o["response_parsed_boxed"], p["answer_gold"]))
            if o["is_unknown"]:
                ok = False
            cat = CAT.get(p["question_type"], p["question_type"])
            hit_cap = o["usage"]["completion_tokens"] >= 20000
            for k in (cat, "ALL_rule_based", f"{dom_dir.name}:all"):
                tot[k][0] += ok; tot[k][1] += 1
            capped["hit_20k_cap" if hit_cap else "under_cap"][0] += ok
            capped["hit_20k_cap" if hit_cap else "under_cap"][1] += 1
            out_rows.append({"question_id": qid, "domain": dom_dir.name, "question_type": p["question_type"],
                             "category": cat, "score_bool": bool(ok), "is_unknown": o["is_unknown"],
                             "completion_tokens": o["usage"]["completion_tokens"]})
        with open(dom_dir / "partial_rule_scores.jsonl", "w") as fp:
            for r in out_rows: fp.write(json.dumps(r) + "\n")
    print(f"skipped (LLM-judged): {skipped}")
    for k in sorted(tot): print(f"{k:28s} {tot[k][0]:4d}/{tot[k][1]:<4d} = {tot[k][0]/tot[k][1]:.3f}")
    for k in sorted(capped): print(f"{k:28s} {capped[k][0]:4d}/{capped[k][1]:<4d} = {capped[k][0]/max(capped[k][1],1):.3f}")

if __name__ == "__main__":
    main(sys.argv[1])
