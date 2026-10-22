#!/usr/bin/env python3
"""Split an E2 arm's reader run into prioritised phases (separate output folders, so the harness's
cached-prompt check — prompt_rows ids must equal the requested ids — holds for each phase).

  p1: complete questions whose answer-phrase coverage of the arm's raw blocks differs from B1's
      blocks (gain or loss)                      -> early, valid R4 (layer 4 + paired test)
  p2: LLM-judged questions (false-premise + gotchas) not in p1
  p3: everything else
For each domain and phase: <runs>/<arm>_<phase>_small/<dom>_small/{prompt_rows.jsonl (subset of the
arm's prompt rows), reader_outputs_cache.jsonl (already cached outputs of these ids), ids.txt}.
Written once; existing phase folders are not overwritten.
Usage: python -I e1/pipeline/split_phases.py --arm e2_single --ret e2_single
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.analysis.retrieval_eval import Content, label_question  # noqa: E402
from e1.common.io import iter_jsonl  # noqa: E402
from e1.pipeline.retrieve import meta  # noqa: E402

import re
QID = re.compile(r'"question_id": "([^"]+)"')
E1 = Path("~/e1").expanduser()
LLM = {"llm_abstention_checker", "llm_gotchas_checker"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="e2_single")
    ap.add_argument("--ret", default="e2_single")
    a = ap.parse_args()
    for dom in ["web", "enterprise"]:
        M, C = meta(dom), Content(dom)
        Q = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl")}
        R = {r["question_id"]: r for r in iter_jsonl(E1 / f"e2work/retrieved/{a.ret}/{dom}.jsonl")}
        C.load({b["entry_id"] for q in R for b in R[q]["raw_blocks"] + Q[q]["baseline_raw_blocks"]})
        p1, p2, p3 = [], [], []
        for q in sorted(R):
            complete = not (Q[q]["context_truncated"] or Q[q]["missing_detected"])
            lx = label_question(M[q], [b["entry_id"] for b in R[q]["raw_blocks"]], C)
            changed = False
            if complete and "ans" in lx:
                lb = label_question(M[q], [b["entry_id"] for b in Q[q]["baseline_raw_blocks"]], C)
                changed = lx["ans"] != lb["ans"]
            if changed:
                p1.append(q)
            elif M[q]["eval_function"].split("|")[0] in LLM:
                p2.append(q)
            else:
                p3.append(q)
        src = E1 / f"runs/{a.arm}_small/{dom}_small"
        cache = {}
        cp = src / "reader_outputs_cache.jsonl"
        if cp.exists():
            for line in open(cp, encoding="utf-8", errors="replace"):
                try:
                    c = json.loads(line)
                    if isinstance(c.get("output"), dict):
                        cache[c["question_id"]] = line if line.endswith("\n") else line + "\n"
                except json.JSONDecodeError:
                    pass
        for name, ids in [("p1", p1), ("p2", p2), ("p3", p3)]:
            dst = E1 / f"runs/{a.arm}_{name}_small/{dom}_small"
            if (dst / "prompt_rows.jsonl").exists():
                print(f"{dst} exists, skipped"); continue
            dst.mkdir(parents=True, exist_ok=True)
            want = set(ids)
            n = 0
            with open(dst / "prompt_rows.jsonl", "w") as fp:
                for line in open(src / "prompt_rows.jsonl"):
                    m = QID.search(line[:5000])
                    if m and m.group(1) in want:
                        fp.write(line); n += 1
            with open(dst / "reader_outputs_cache.jsonl", "w") as fp:
                for q in ids:
                    if q in cache:
                        fp.write(cache[q])
            (dst / "ids.txt").write_text(",".join(ids))
            print(f"{dom} {name}: {len(ids)} questions ({n} prompt rows, {sum(q in cache for q in ids)} already cached) -> {dst}")


if __name__ == "__main__":
    main()
