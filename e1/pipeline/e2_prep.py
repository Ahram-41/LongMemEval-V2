#!/usr/bin/env python3
"""E2 inputs that need no GPU.

  meta    : ~/e1/cache/meta/<dom>.jsonl  question text / type / abstention flag / eval_function /
            gold answer per question, streamed from the baseline per_question.jsonl
  pieces  : ~/e1/index_ext/dense_full/<dom>/pieces.jsonl  every state's accessibility-tree text
            ("State {i}\\n{text}", the per-state part of the official slice_axtree_text) split into
            consecutive pieces of <= --max-tokens tokens of the Qwen3-Embedding-8B tokenizer
            (no special tokens; cut at token offsets so each piece is an exact substring)
Run pieces with the vllm env python (it has `tokenizers`); meta needs only the stdlib.
(Query strings are collected and embedded by e2_gpu.py queries.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.common.manifest import write_manifest  # noqa: E402

E1 = Path("~/e1").expanduser()
TOKENIZER = E1 / "hf_cache/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/tokenizer.json"
META_KEEP = ("question_id", "question_type", "is_abstention_problem", "eval_function", "question_text", "answer_gold")


def do_meta(dom: str) -> None:
    out = E1 / "cache/meta"; out.mkdir(parents=True, exist_ok=True)
    src = E1 / f"runs/agentrunbook_r_small/{dom}_small/per_question.jsonl"
    with open(out / f"{dom}.jsonl", "w") as fp:
        for r in iter_jsonl(src):
            fp.write(json.dumps({k: r.get(k) for k in META_KEEP}, ensure_ascii=True) + "\n")
            del r
    print(f"meta {dom} -> {out / f'{dom}.jsonl'}")


def state_texts(dom: str, limit_traj: int | None = None):
    ms = E1 / f"index/agentrunbook_r_{dom}_small/memory_state"
    tids = json.loads((ms / "index.json").read_text())["inserted_trajectory_ids"]
    for tid in tids[:limit_traj] if limit_traj else tids:
        t = json.loads((ms / "trajectories" / tid / "trajectory.json").read_text())
        for s in t["states"]:
            text = s.get("text") if isinstance(s.get("text"), str) else ""
            yield tid, int(s["state_index"]), "\n".join([f"State {s['state_index']}", text])


def do_pieces(dom: str, max_tokens: int, limit_traj: int | None, out_name: str) -> None:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    out = E1 / "index_ext/dense_full" / dom; out.mkdir(parents=True, exist_ok=True)
    n_states = n_pieces = n_tok = 0
    with open(out / out_name, "w") as fp:
        for tid, si, text in state_texts(dom, limit_traj):
            enc = tok.encode(text, add_special_tokens=False)
            offs = enc.offsets
            n = len(offs)
            n_states += 1; n_tok += n
            starts = list(range(0, max(n, 1), max_tokens))
            for k, a in enumerate(starts):
                b = min(n, a + max_tokens)
                if n == 0:
                    c0, c1 = 0, len(text)
                else:
                    c0 = 0 if k == 0 else offs[a][0]
                    c1 = len(text) if b == n else offs[b][0]
                fp.write(json.dumps({"piece_id": n_pieces, "trajectory_id": tid, "state_index": si, "piece": k,
                                     "n_tokens": b - a, "text": text[c0:c1]}, ensure_ascii=True) + "\n")
                n_pieces += 1
    print(f"pieces {dom}: {n_states} states, {n_pieces} pieces, {n_tok} tokens -> {out / out_name}")
    write_manifest(out, inputs={"tokenizer": TOKENIZER}, params={"max_tokens": max_tokens, "limit_traj": limit_traj},
                   name=f"manifest_{out_name}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["meta", "pieces"])
    ap.add_argument("--domains", default="web,enterprise")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit-traj", type=int, default=None)
    ap.add_argument("--out-name", default="pieces.jsonl")
    a = ap.parse_args()
    for dom in a.domains.split(","):
        if a.what == "meta":
            do_meta(dom)
        else:
            do_pieces(dom, a.max_tokens, a.limit_traj, a.out_name)


if __name__ == "__main__":
    main()
