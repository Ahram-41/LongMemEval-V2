"""Lean readers for E1/E2 run folders.

per_question.jsonl rows carry the full memory_context (often >1 MB), and the login node has a
~1 GB ulimit, so every reader streams line by line and keeps only the requested fields.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

DOMAINS = ("web_small", "enterprise_small")
SCORE_KEEP = ("question_id", "score_bool", "is_unknown", "is_abstention_problem", "question_type",
              "usage", "memory_query_duration_seconds")
CAP = 20000


def iter_jsonl(path: Path | str) -> Iterator[dict]:
    with open(path) as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def load_rows(run_root: Path, arm: str, keep: Iterable[str] = SCORE_KEEP, fname: str = "per_question.jsonl",
              require_all: bool = False, domains: Iterable[str] = DOMAINS) -> dict | None:
    """{question_id: {k: row[k] for k in keep}} over both domains.

    require_all=True returns None as soon as one domain file is missing (compare_arms semantics);
    otherwise missing domain files are skipped."""
    keep = tuple(keep)
    rows: dict = {}
    for d in domains:
        p = Path(run_root) / arm / d / fname
        if not p.exists():
            if require_all:
                return None
            continue
        for r in iter_jsonl(p):
            rows[r["question_id"]] = {k: r.get(k) for k in keep}
            del r
    return rows


def capped(row: dict, cap: int = CAP) -> bool:
    return (row.get("usage") or {}).get("completion_tokens", 0) >= cap
