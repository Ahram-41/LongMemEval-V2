"""Load the official AgentRunbook-R memory from a saved memory_state, unchanged.

On CPU nodes transformers 5.x refuses to import ("Could not import module 'AutoTokenizer'"),
so with cpu=True a stub `transformers` module is installed before import. The stub only
replaces AutoTokenizer, which the official code touches lazily (_get_embedding_tokenizer) and
which rendering, pools and merge never call; any call to it raises.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
E1 = Path("~/e1").expanduser()


def _stub_transformers() -> None:
    if "transformers" in sys.modules:
        return
    mod = types.ModuleType("transformers")

    class _NoTokenizer:
        @staticmethod
        def from_pretrained(*_a, **_k):
            raise RuntimeError("transformers stubbed on a CPU node: tokenizer unavailable")

    mod.AutoTokenizer = _NoTokenizer

    def __getattr__(name):  # anything else imported at module level
        return type(name, (), {"from_pretrained": _NoTokenizer.from_pretrained})
    mod.__getattr__ = __getattr__
    sys.modules["transformers"] = mod


def memory_state_dir(dom: str) -> Path:
    return E1 / f"index/agentrunbook_r_{dom}_small/memory_state"


def load_official(dom: str, cpu: bool = False, runs_arm: str = "agentrunbook_r_small"):
    """Official AgentRunbookR loaded exactly as the harness does (load_memory with the run's
    runtime memory_config.json as the requested config); base URLs follow the job's env."""
    if cpu:
        _stub_transformers()
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from memory_modules.memory import load_memory
    cfg = json.loads((E1 / f"runs/{runs_arm}/{dom}_small/runtime_inputs/memory_config.json").read_text())
    p = cfg["memory_params"]
    if os.environ.get("LME_CONTROLLER_BASE_URL"):
        p["controller_params"]["base_url"] = os.environ["LME_CONTROLLER_BASE_URL"]
    if os.environ.get("LME_EMBEDDING_BASE_URL"):
        p["embedding_params"]["base_url"] = os.environ["LME_EMBEDDING_BASE_URL"]
    return load_memory(memory_state_dir(dom), requested_config=cfg)
