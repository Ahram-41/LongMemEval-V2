"""Shared paths and formatting for the STALE port (mem-test v2 protocol on the cluster).

Port of CP5101/mem-test/stale_gen_v2.py + stale_judge_v2.py (local MLX) to OpenAI-compatible endpoints.
Data: HF STALEproj/STALE rev 617c51dc, T1_T2_400_FULL.json (sha256 5f3ec375...1c876), under ~/e1/data/stale.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

E1 = Path("~/e1").expanduser()
DATA = E1 / "data/stale/T1_T2_400_FULL.json"
WORK = Path(os.environ.get("STALE_WORK", E1 / "stalework"))
PREP = WORK / "prep"
RET = WORK / "retrieval"
RUNS = WORK / "runs"
DIMS = ("dim1_query", "dim2_query", "dim3_query")
CONDS = ("C0", "M_stale", "S_rm_current", "O_both", "O_new_only")
K = 10


def fmt(u: dict) -> str:
    """Unit text exactly as mem-test v2 embedded and showed it."""
    return f"[{u['date']}] {u['role']}: {u['text']}"


def iter_jsonl(path):
    with open(path) as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def load_scenarios(name: str) -> list[dict]:
    return list(iter_jsonl(PREP / f"{name}.jsonl"))


def client(base_url: str, api_key: str = "EMPTY", timeout: float = 600):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
