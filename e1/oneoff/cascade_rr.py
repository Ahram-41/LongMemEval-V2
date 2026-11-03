#!/usr/bin/env python3
"""Offline check of a rerank-on-demand cascade (2026-11-03), from existing reader outputs only.

First pass = iv_base (official retrieval, strict + forced wrap-up, 295 rule-scored questions).
Second pass = reranked retrieval with forced wrap-up, 295 = rrw_sX + c2rr_sX (two samples).
Triggers (both visible at test time, no gold needed):
  T1  first-pass final answer (after forced wrap-up for capped rows) is UNKNOWN
  T2  first pass said UNKNOWN or hit the 20k cap (before wrap-up)
Cascade answer = second pass if triggered, else first pass. Scoring = score_uniform.load.
Also counts how often T1/T2 would fire on the judge-graded questions (false premise, gotchas) in b1_small,
where no reranked run exists.

  python -I e1/oneoff/cascade_rr.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.oneoff.reader_interventions import mcnemar  # noqa: E402
from e1.oneoff.score_uniform import DOMS, E1, combine  # noqa: E402


def is_unk(s) -> bool:
    return str(s or "").strip().strip(".").lower() == "unknown"


def first_pass(v: str = "base") -> dict:
    """question -> {unk_final, unk_or_cap, type}"""
    out = {}
    for dom in DOMS:
        d = E1 / f"runs/iv_{v}_small/{dom}_small"
        w2 = {}
        if (d / "wrap2.jsonl").exists():
            w2 = {json.loads(l)["question_id"]: json.loads(l) for l in open(d / "wrap2.jsonl") if l.strip()}
        for line in open(d / "outputs.jsonl"):
            o = json.loads(line)
            q, cap = o["question_id"], o["cap_no_answer"]
            final = o.get("wrap2_boxed", w2.get(q, {}).get("wrap2_boxed")) if cap else o["boxed"]
            out[q] = {"unk_final": is_unk(final), "unk_or_cap": cap or is_unk(o["boxed"]),
                      "type": o["question_type"].split("-")[0]}
    return out


def main() -> None:
    fp = first_pass()
    base = combine("base")
    qs = sorted(fp)
    assert len(qs) == 295 and set(qs) == set(base), (len(qs), len(base))
    print(f"first pass (iv_base): correct {sum(base[q]['ok'] for q in qs)}/295")
    for trig in ("unk_final", "unk_or_cap"):
        T = [q for q in qs if fp[q][trig]]
        print(f"\n== trigger {trig}: fires on {len(T)}/295  (first pass right on {sum(base[q]['ok'] for q in T)} of them)"
              f"  by type {dict(Counter(fp[q]['type'] for q in T))}")
        for s in ("s1", "s2"):
            rr = combine(f"rrw_{s}+c2rr_{s}")
            assert set(rr) == set(qs)
            fix = [q for q in T if rr[q]["ok"] and not base[q]["ok"]]
            brk = [q for q in T if base[q]["ok"] and not rr[q]["ok"]]
            casc = sum((rr if fp[q][trig] else base)[q]["ok"] for q in qs)
            N = [q for q in qs if not fp[q][trig]]
            nfix = sum(rr[q]["ok"] and not base[q]["ok"] for q in N)
            nbrk = sum(base[q]["ok"] and not rr[q]["ok"] for q in N)
            full = sum(rr[q]["ok"] for q in qs)
            print(f"  rerank {s}: triggered -> rerank right {sum(rr[q]['ok'] for q in T)}/{len(T)}, "
                  f"+{len(fix)}/-{len(brk)} (p={mcnemar(len(fix), len(brk)):.3f})  | not triggered, had we "
                  f"reranked anyway: +{nfix}/-{nbrk}")
            print(f"     cascade {casc}/295 ({casc - 219:+d} vs 219)   rerank-everything {full}/295   fixed: {fix}"
                  f"   broken: {brk}")
            by = Counter((fp[q]["type"], "fix") for q in fix) + Counter((fp[q]["type"], "break") for q in brk)
            print(f"     by type {dict(by)}")

    # judge-graded questions: how often would the trigger fire, and how many of those are judged right now
    print("\n== judge-graded questions in b1_small (no reranked run exists for them)")
    rows = [json.loads(l) for dom in DOMS for l in open(E1 / f"runs/b1_small/{dom}_small/per_question.jsonl")]
    for name, sel in (("false premise", lambda r: r["is_abstention_problem"]),
                      ("gotchas", lambda r: (not r["is_abstention_problem"]) and r["category"] == "gotchas")):
        R = [r for r in rows if sel(r)]
        cap = [r for r in R if r["usage"]["completion_tokens"] >= 20000]
        unk = [r for r in R if r["is_unknown"] and r["usage"]["completion_tokens"] < 20000]
        sc = lambda rs: sum(bool(r.get("score_bool")) for r in rs)
        print(f"  {name}: n={len(R)} judged right {sc(R)} | UNKNOWN (not capped) {len(unk)}, judged right {sc(unk)}"
              f" | capped {len(cap)}, judged right {sc(cap)}")


if __name__ == "__main__":
    main()
