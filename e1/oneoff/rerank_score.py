"""Score (query, document) pairs with Qwen3-Reranker-8B under vLLM (offline LLM, runs in the vllm env).

Official Qwen3-Reranker recipe: chat-formatted prompt, one generated token, score = P(yes) over {yes, no}.
Reads rerank/pairs_<dom>.jsonl, writes rerank/scores_<dom>.jsonl (question_id, query, entry_id, p_yes).
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

E1 = Path(os.environ.get("E1", "~/e1")).expanduser()
MODEL = os.environ.get("RERANKER", str(E1 / "models/Qwen3-Reranker-8B"))
INSTRUCT = ("Given a search query written by a web agent that describes a web page or UI state it is looking for, "
            "judge whether the Document (an accessibility-tree snapshot of one page state) is the page state the query describes "
            "and contains the information the query asks about.")
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def fmt(query: str, doc: str) -> str:
    return f"{PREFIX}<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>: {doc}{SUFFIX}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="web,enterprise")
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--gpu-frac", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_frac,
              enable_prefix_caching=True, max_num_seqs=64)
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=2, allowed_token_ids=[yes, no])
    work = E1 / "e2work/rerank"
    for dom in args.domains.split(","):
        rows = [json.loads(l) for l in open(work / f"pairs_{dom}.jsonl")]
        if args.limit:
            rows = rows[: args.limit]
        t0 = time.time()
        prompts = []
        for r in rows:
            ids = tok(fmt(r["query"], r["doc"]), add_special_tokens=False)["input_ids"]
            if len(ids) > args.max_model_len - 8:  # cut the document, keep the suffix intact
                head = tok(fmt(r["query"], ""), add_special_tokens=False)["input_ids"]
                budget = args.max_model_len - 8 - len(head)
                doc_ids = tok(r["doc"], add_special_tokens=False)["input_ids"][:budget]
                prompts.append(fmt(r["query"], tok.decode(doc_ids)))
            else:
                prompts.append(fmt(r["query"], r["doc"]))
        outs = llm.generate(prompts, sp, use_tqdm=False)
        with open(work / f"scores_{dom}.jsonl", "w") as fp:
            for r, o in zip(rows, outs):
                lp = o.outputs[0].logprobs[0]
                ly = lp[yes].logprob if yes in lp else -1e9
                ln = lp[no].logprob if no in lp else -1e9
                p = 1.0 / (1.0 + math.exp(ln - ly))
                fp.write(json.dumps({"question_id": r["question_id"], "query": r["query"], "entry_id": r["entry_id"],
                                     "piece_id": r["piece_id"], "p_yes": p}) + "\n")
        print(f"[rerank] {dom}: {len(rows)} pairs in {time.time() - t0:.0f}s -> scores_{dom}.jsonl", flush=True)


if __name__ == "__main__":
    main()
