#!/usr/bin/env python3
"""Build a full 451-question arm from B1 plus an arm that reran only noted questions.

Usage: python e1/pipeline/merge_noted.py ~/e1/runs b1_small pa_small pa_merged_small
For each domain: rows = B1 per_question.jsonl, with every question present in the
noted arm's per_question.jsonl replaced by the noted arm's row. Valid because the
unnoted prompts are byte-identical to B1's (only the system prompt fix is applied).
"""
import json, sys
from pathlib import Path

root, b1, noted, out = Path(sys.argv[1]).expanduser(), sys.argv[2], sys.argv[3], sys.argv[4]
for d in ["web_small", "enterprise_small"]:
    base = {json.loads(l)["question_id"]: json.loads(l) for l in open(root / b1 / d / "per_question.jsonl")}
    p = root / noted / d / "per_question.jsonl"
    repl = {json.loads(l)["question_id"]: json.loads(l) for l in open(p)} if p.exists() else {}
    assert set(repl) <= set(base), "noted arm has questions B1 lacks"
    (root / out / d).mkdir(parents=True, exist_ok=True)
    with open(root / out / d / "per_question.jsonl", "w") as fp:
        for q, r in base.items():
            fp.write(json.dumps(repl.get(q, r)) + "\n")
    print(f"{d}: {len(base)} rows, {len(repl)} replaced from {noted}")
