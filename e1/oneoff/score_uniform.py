#!/usr/bin/env python3
"""One strict + forced-wrap-up rule for every iv_* arm, including iv_base (2026-11-03).

Rule (as trunc_c.py score): an answer that ended at the 20k cap counts by its forced wrap-up answer
(wrap2_boxed; the prefix ends with "\\boxed{"), otherwise an explicit \\boxed{} in the answer text is required.
iv_base keeps its forced wrap-up in wrap2.jsonl next to outputs.jsonl; the --src arms carry it inline.
(e2work/rerank/stack_score.py scored capped --src rows by the absent wrap_text, i.e. always wrong, and iv_base
by its older free-form wrap-up, so its strict column is not this rule.)

  python -I e1/oneoff/score_uniform.py --arms "STACK_s1=rrw_s1+c2rc_s1,..." [--ref base | --ref "K_s1=base@U+..."] [--n 295]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.oneoff.reader_interventions import correct, correct_strict, mcnemar  # noqa: E402

E1 = Path(os.environ.get("E1", "~/e1")).expanduser()
DOMS = ("web", "enterprise")


def ids(p: Path) -> set[str]:
    return set(p.read_text().replace(",", " ").split())


def load(v: str) -> dict:
    out = {}
    for dom in DOMS:
        d = E1 / f"runs/iv_{v}_small/{dom}_small"
        if not (d / "outputs.jsonl").exists():
            continue
        w2 = {}
        if (d / "wrap2.jsonl").exists():
            w2 = {json.loads(l)["question_id"]: json.loads(l) for l in open(d / "wrap2.jsonl") if l.strip()}
        for line in open(d / "outputs.jsonl"):
            o = json.loads(line)
            q = o["question_id"]
            if o["cap_no_answer"]:
                b = o.get("wrap2_boxed", w2.get(q, {}).get("wrap2_boxed"))
                if b is None:
                    raise SystemExit(f"iv_{v} {dom} {q}: capped with no forced wrap-up")
                ok = bool(correct(b, o))
            else:
                ok = correct_strict(o["content"], o)
            out[q] = {"ok": ok, "cap": o["cap_no_answer"], "dom": dom}
    return out


def combine(spec: str) -> dict:
    """v1+v2+...; v@S keeps only the questions in ~/e1/v2work/sys_S_ids.txt."""
    d = {}
    for p in spec.split("+"):
        v, _, sub = p.partition("@")
        part = load(v)
        if sub:
            keep = ids(E1 / f"v2work/sys_{sub}_ids.txt")
            part = {q: r for q, r in part.items() if q in keep}
        dup = set(d) & set(part)
        if dup:
            raise SystemExit(f"{spec}: {len(dup)} questions in more than one part")
        d.update(part)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="NAME=v1+v2,...")
    ap.add_argument("--ref", default="base", help="reference: an iv variant, or NAME=v1+v2")
    ap.add_argument("--n", type=int, default=295)
    a = ap.parse_args()
    ref = combine(a.ref.split("=", 1)[1] if "=" in a.ref else a.ref)
    AW = ids(E1 / "e2work/rerank/ids_always_wrong.txt")
    AR = ids(E1 / "e2work/rerank/ids_always_right.txt")
    print(f"ref {a.ref}: {len(ref)} questions, correct {sum(r['ok'] for r in ref.values())}, caps {sum(r['cap'] for r in ref.values())}")
    for spec in a.arms.split(","):
        name, parts = spec.split("=", 1)
        A = combine(parts)
        qs = sorted(set(A) & set(ref))
        if len(qs) != a.n:
            print(f"  [warn] {name}: {len(qs)} paired questions, expected {a.n}")
        for label, sub in (("all", None), ("always_wrong", AW), ("always_right", AR)):
            s = [q for q in qs if sub is None or q in sub]
            b = sum(A[q]["ok"] and not ref[q]["ok"] for q in s)
            c = sum(ref[q]["ok"] and not A[q]["ok"] for q in s)
            ra, aa = sum(ref[q]["ok"] for q in s), sum(A[q]["ok"] for q in s)
            print(f"  {name:10} {label:12} n={len(s):3} ref {ra:3} -> {aa:3}  +{b}/-{c}  net {aa - ra:+d}  "
                  f"p={mcnemar(b, c):.3f}  caps {sum(ref[q]['cap'] for q in s)} -> {sum(A[q]['cap'] for q in s)}")


if __name__ == "__main__":
    main()
