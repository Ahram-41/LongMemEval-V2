"""Per-question table for the oracle arm: B1 vs orc rule scores joined with oracle/plan.json.
Usage: python -I e1/oneoff/oracle_report.py [--arm orc_small]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.analysis.rule_compare import E1, arm_outputs  # noqa: E402
from e1.common.io import iter_jsonl  # noqa: E402


def main() -> None:
    from evaluation.qa_eval_metrics import eval_from_spec, score_to_bool
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="orc_small")
    a = ap.parse_args()
    plan = json.load(open(E1 / "e2work/oracle/plan.json"))
    rows = []
    for dom in ("web", "enterprise"):
        b1 = {r["question_id"]: r for r in iter_jsonl(E1 / f"runs/b1_small/{dom}_small/per_question.jsonl")}
        arm = arm_outputs(a.arm, dom)
        for q, p in plan.items():
            if p["domain"] != dom:
                continue
            r = b1[q]
            o = arm.get(q)
            if o is None:
                rows.append((q, p, bool(r["score_bool"]), None, None, None))
                continue
            boxed = o["response_parsed_boxed"]
            ok = (not o["is_unknown"]) and score_to_bool(eval_from_spec(r["eval_function"], boxed, r["answer_gold"]))
            rows.append((q, p, bool(r["score_bool"]), bool(ok), boxed, f"out={o['usage']['completion_tokens']}"))
    c = Counter()
    for q, p, b, oc, boxed, fin in rows:
        tag = f"{p['source']:8} B1 {'ok ' if b else 'WRONG'} -> orc {'?' if oc is None else ('ok' if oc else 'wrong')}"
        c[(p["source"], b, oc)] += 1
        print(f"{q} {p['domain'][:3]} {tag:40} inserted {[(e[:13], g) for e, g, s in p['inserted']]} boxed={str(boxed)[:40]!r} {fin or ''}")
    print()
    for k, v in sorted(c.items(), key=str):
        print(f"  source={k[0]:8} b1_correct={k[1]!s:5} orc_correct={k[2]!s:5} n={v}")
    miss = [q for q, p, b, oc, _, _ in rows if p["source"] == "evidence"]
    fixed = [q for q, p, b, oc, _, _ in rows if p["source"] == "evidence" and oc]
    print(f"\n13 true misses (evidence rules): oracle correct {len(fixed)}/{len(miss)}: {fixed}")
    broke = [q for q, p, b, oc, _, _ in rows if b and oc is False]
    print(f"B1-correct questions broken by the oracle insert: {len(broke)}/{sum(1 for r in rows if r[2])}: {broke}")


if __name__ == "__main__":
    main()
