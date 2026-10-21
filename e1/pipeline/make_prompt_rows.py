#!/usr/bin/env python3
"""E2: prompt rows for a retrieval arm, rerunning the reader only where retrieval changed.

For each question: if its raw blocks (entry, query, rank sequence) equal the blocks B1 actually
used (B0 context; untruncated, parsed queries) the B1 reader output is reused and no row is
written. Otherwise (blocks differ, or queries were regenerated) a prompt row is written: B1's row
with the new memory context (contexts.jsonl from build_context.py), passed through the official
truncate_memory_context (200k) and build_messages with B1's fixed system prompt.
Writes <dst>/<dom>_small/prompt_rows.jsonl, rerun_ids.txt and reuse_ids.txt.
Needs a GPU node (the harness imports transformers).
Usage: python e1/pipeline/make_prompt_rows.py --contexts <dir> --retrieved <dir> --dst ~/e1/runs/e2_<arm>_small
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.common.manifest import write_manifest  # noqa: E402

E1 = Path("~/e1").expanduser()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True, help="dir with <dom>_contexts.jsonl")
    ap.add_argument("--retrieved", required=True, help="dir with <dom>.jsonl from retrieve.py select")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--b1", default=str(E1 / "runs/b1_small"))
    ap.add_argument("--domains", default="web,enterprise")
    ap.add_argument("--memory-context-max-tokens", type=int, default=200000)
    ap.add_argument("--only", choices=["all", "regenerated"], default="all",
                    help="regenerated: write rows only for the 107 questions whose B0 context was truncated")
    ap.add_argument("--section-order", default=None,
                    help="e.g. raw,events,notes: reorder memory sections before the 200k truncation (E2 arm A)")
    ap.add_argument("--force-rerun", action="store_true", help="rerun every selected question even if blocks equal B1's")
    ap.add_argument("--ids-file", default=None, help="restrict to these question ids (comma-separated file)")
    a = ap.parse_args()
    from evaluation.harness import build_messages, truncate_memory_context
    keep_ids = {x for x in Path(a.ids_file).expanduser().read_text().split(",") if x} if a.ids_file else None
    for dom in a.domains.split(","):
        dst = Path(a.dst).expanduser() / f"{dom}_small"
        dst.mkdir(parents=True, exist_ok=True)
        Q = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl")}
        R = {r["question_id"]: r for r in iter_jsonl(Path(a.retrieved).expanduser() / f"{dom}.jsonl")}
        rerun, reuse, reasons = [], [], {}
        for qid, r in R.items():
            b0 = [(b["entry_id"], b["query"], b["rank"]) for b in Q[qid]["baseline_raw_blocks"]]
            new = [(b["entry_id"], b["query"], b["rank"]) for b in r["raw_blocks"]]
            if a.only == "regenerated" and r["query_source"] != "regenerated":
                continue
            if a.ids_file and qid not in keep_ids:
                continue
            same = r["query_source"] == "parsed" and not Q[qid]["context_truncated"] and b0 == new and not a.force_rerun
            (reuse if same else rerun).append(qid)
            reasons[qid] = {"question_id": qid, "rerun": not same,
                            "reason": ("reuse: identical (entry, query, rank) sequence" if same else
                                       "regenerated queries (B0 context truncated / query missing)" if r["query_source"] == "regenerated"
                                       else "same block set, different order/rank" if {x[0] for x in b0} == {x[0] for x in new}
                                       else "different blocks"),
                            "b1_blocks": [x[0] for x in b0], "arm_blocks": [x[0] for x in new],
                            "n_shared_blocks": len({x[0] for x in b0} & {x[0] for x in new})}
        want = set(rerun)
        ctx = {}
        for c in iter_jsonl(Path(a.contexts).expanduser() / f"{dom}_contexts.jsonl"):
            if c["question_id"] in want:
                ctx[c["question_id"]] = c["memory_context"]
        n = 0
        with open(dst / "prompt_rows.jsonl", "w") as fp:
            for row in iter_jsonl(Path(a.b1).expanduser() / f"{dom}_small" / "prompt_rows.jsonl"):
                qid = row["question_id"]
                if qid not in want:
                    del row
                    continue
                mc0 = ctx[qid]
                if a.section_order:
                    from e1.pipeline.derive_arm import split_sections
                    secs = split_sections(mc0)
                    mc0 = [it for name in a.section_order.split(",") for it in secs.get(name, [])]
                mc, orig, trunc = truncate_memory_context(mc0, max_tokens=a.memory_context_max_tokens, question_id=qid)
                system_prompt = next(m["content"] for m in row["messages"] if m["role"] == "system")
                row["messages"], row["prompt_messages"] = build_messages(
                    system_prompt=system_prompt, question_text=row["question_text"],
                    image_path=row["question_image"], memory_context=mc)
                row["memory_context"] = mc
                row["memory_context_original_token_count"] = orig
                row["memory_context_token_count"] = trunc
                row["memory_context_was_truncated"] = trunc < orig
                row["e2_raw_block_ids"] = [b["entry_id"] for b in R[qid]["raw_blocks"]]
                row["e2_query_source"] = R[qid]["query_source"]
                row["e2_variant"] = R[qid]["variant"] + "@" + R[qid]["query_source_mode"]
                row["e2_section_order"] = a.section_order or "notes,events,raw"
                fp.write(json.dumps(row, ensure_ascii=True) + "\n")
                n += 1
                del row
        assert n == len(rerun), (n, len(rerun))
        with open(dst / "rerun_reasons.jsonl", "w") as rf:
            for qid in sorted(reasons):
                rf.write(json.dumps(reasons[qid]) + "\n")
        (dst / "rerun_ids.txt").write_text(",".join(sorted(rerun)))
        (dst / "reuse_ids.txt").write_text(",".join(sorted(reuse)))
        write_manifest(dst, inputs={"contexts": a.contexts, "retrieved": a.retrieved, "b1": a.b1},
                       params={"memory_context_max_tokens": a.memory_context_max_tokens, "only": a.only,
                               "section_order": a.section_order, "force_rerun": a.force_rerun,
                               "ids_file": a.ids_file})
        print(f"[prompt_rows] {dom}: rerun {len(rerun)} (regenerated queries {sum(R[q]['query_source'] == 'regenerated' for q in rerun)}), "
              f"reuse B1 {len(reuse)} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
