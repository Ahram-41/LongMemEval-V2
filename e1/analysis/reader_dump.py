#!/usr/bin/env python3
"""Compact per-question reader record for an arm (E2 logging requirement 4), streamed so it runs on
the login node: rerun flag and reason, prompt/context token counts, truncation flag, completion
tokens, cap hit, boxed answer, UNKNOWN flag, judge score. Rows of a merged arm come from the
rerun arm when rerun, else from B1.
Usage: python3 e1/analysis/reader_dump.py ~/e1/runs e2_x_small b1_small > reader_e2_x.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import CAP, DOMAINS, iter_jsonl  # noqa: E402

root, arm, b1 = Path(sys.argv[1]).expanduser(), sys.argv[2], sys.argv[3]
KEEP = ("question_id", "question_type", "is_abstention_problem", "memory_context_original_token_count",
        "memory_context_token_count", "memory_context_was_truncated", "usage", "response_parsed_boxed",
        "is_unknown", "score", "score_bool", "answer_gold")
for d in DOMAINS:
    reasons = {}
    rp = root / arm / d / "rerun_reasons.jsonl"
    if rp.exists():
        reasons = {r["question_id"]: r for r in iter_jsonl(rp)}
    rows = {}
    for src in [b1, arm]:
        p = root / src / d / "per_question.jsonl"
        if p.exists():
            for r in iter_jsonl(p):
                rows[r["question_id"]] = ({k: r.get(k) for k in KEEP}, src)
                del r
    for q, (r, src) in sorted(rows.items()):
        u = r.pop("usage") or {}
        out = {"arm": arm, "domain": d, **r, "from_run": src, "prompt_tokens": u.get("prompt_tokens"),
               "completion_tokens": u.get("completion_tokens"), "cap_hit": (u.get("completion_tokens") or 0) >= CAP,
               "rerun": reasons.get(q, {}).get("rerun", src == arm), "rerun_reason": reasons.get(q, {}).get("reason")}
        print(json.dumps(out, ensure_ascii=True))
