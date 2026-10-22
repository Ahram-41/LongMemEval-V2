#!/usr/bin/env python3
"""Fixed dev / held-out split of LongMemEval_S questions (written before any retrieval result).

Rule: h = int(sha256(question_id.encode()).hexdigest()[:8], 16); dev if h % 2 == 0, else held-out.
No stratification, no seed. Writes ~/e1/e3work/split/{dev_ids.txt,heldout_ids.txt,split.json}.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.common.manifest import write_manifest  # noqa: E402
from e1.lmes.prep import PREP, WORK  # noqa: E402

RULE = "dev iff int(sha256(question_id).hexdigest()[:8], 16) % 2 == 0"


def is_dev(qid: str) -> bool:
    return int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:8], 16) % 2 == 0


def load_split() -> dict[str, str]:
    d = WORK / "split"
    out = {}
    for name in ("dev", "heldout"):
        for line in open(d / f"{name}_ids.txt"):
            if line.strip():
                out[line.strip()] = name
    return out


def main() -> None:
    rows = [(r["question_id"], r["question_type"], r["is_abs"]) for r in iter_jsonl(PREP / "questions.jsonl")]
    d = WORK / "split"
    d.mkdir(parents=True, exist_ok=True)
    dev = sorted(q for q, _, _ in rows if is_dev(q))
    held = sorted(q for q, _, _ in rows if not is_dev(q))
    (d / "dev_ids.txt").write_text("\n".join(dev) + "\n")
    (d / "heldout_ids.txt").write_text("\n".join(held) + "\n")
    cnt = collections.defaultdict(lambda: [0, 0])
    for q, t, a in rows:
        cnt[t + ("_abs" if a else "")][0 if is_dev(q) else 1] += 1
    summary = {"rule": RULE, "n_dev": len(dev), "n_heldout": len(held), "by_type_dev_heldout": dict(sorted(cnt.items()))}
    (d / "split.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(d, inputs={"questions": PREP / "questions.jsonl"}, params={"rule": RULE})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
