"""Record what produced an arm folder: code version, command line, inputs."""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git_version() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "e1"))}


def write_manifest(dst: Path | str, *, inputs: dict, params: dict | None = None, name: str = "manifest.json") -> Path:
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    m = {"created": _dt.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv, "git": git_version(),
         "inputs": {k: str(v) for k, v in inputs.items()}, "params": params or {}}
    p = dst / name
    p.write_text(json.dumps(m, indent=2) + "\n")
    return p
