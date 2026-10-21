#!/usr/bin/env python3
"""Classify abstention (false-premise) outcomes in a finished run, no GPU or API.

Buckets for each wrong abstention answer:
  unknown_with_cue   boxed UNKNOWN, and the visible response text contains an absence/premise cue
  unknown_no_cue     boxed UNKNOWN, no cue
  capped             hit the 20k completion cap (answer usually truncated / malformed)
  answered_with_cue  concrete answer accepting the premise, but the text contains an absence cue
  answered_no_cue    concrete answer accepting the premise, no cue
The cue regex is a heuristic; its false-positive rate is estimated on correct non-abstention answers.
Writes <run>/abstention_diagnosis.jsonl and prints counts.
"""
import json, re, sys
from collections import Counter
from pathlib import Path

CUE = re.compile(r"does ?n[o']t exist|not exist|no such|not present|there (is|are) no|"
                 r"is ?n[o']t (shown|visible|listed|available|mentioned|present|there)|"
                 r"does ?n[o']t (contain|appear|show|have|mention|confirm|include|support)|"
                 r"not found|no evidence|cannot find|can't find|could not find|premise|flawed|"
                 r"false assumption|incorrect assumption|not (supported|possible|achievable)|impossible",
                 re.I)
CAP = 20000

def bucket(r):
    raw = r["response_raw"] or ""; cue = bool(CUE.search(raw))
    capped = r["usage"]["completion_tokens"] >= CAP if r.get("usage") else False
    if r["score_bool"]: return "correct", cue, capped
    if r["is_unknown"]: return ("unknown_with_cue" if cue else "unknown_no_cue"), cue, capped
    if capped: return "capped", cue, capped
    return ("answered_with_cue" if cue else "answered_no_cue"), cue, capped

def main(run_root):
    root = Path(run_root); rows = []
    for d in sorted(root.glob("*_small")):
        for line in open(d / "per_question.jsonl"):
            r = json.loads(line); r["_domain"] = d.name; rows.append(r)
    abs_rows = [r for r in rows if r["is_abstention_problem"]]
    non_ok = [r for r in rows if not r["is_abstention_problem"] and r["score_bool"]]
    out = []; c = Counter(); by_type = Counter()
    for r in abs_rows:
        b, cue, capped = bucket(r); c[b] += 1; by_type[(r["question_type"], b)] += 1
        out.append({"question_id": r["question_id"], "domain": r["_domain"], "question_type": r["question_type"],
                    "bucket": b, "cue": cue, "capped": capped,
                    "completion_tokens": r["usage"]["completion_tokens"] if r.get("usage") else None,
                    "boxed": (r["response_parsed_boxed"] or "")[:200], "gold": r["answer_gold"][:200]})
    with open(root / "abstention_diagnosis.jsonl", "w") as fp:
        for o in out: fp.write(json.dumps(o) + "\n")
    n = len(abs_rows); print(f"abstention questions: {n}")
    for k in ["correct", "unknown_with_cue", "unknown_no_cue", "capped", "answered_with_cue", "answered_no_cue"]:
        print(f"  {k:18s} {c[k]:4d}  ({c[k]/n:.3f})")
    fp_rate = sum(bool(CUE.search(r["response_raw"] or "")) for r in non_ok) / max(len(non_ok), 1)
    print(f"cue false-positive rate on correct non-abstention answers: {fp_rate:.3f} (n={len(non_ok)})")
    print("by type:")
    for qt in sorted({r['question_type'] for r in abs_rows}):
        print("  ", qt, {b: by_type[(qt, b)] for b in c if by_type[(qt, b)]})

if __name__ == "__main__":
    main(sys.argv[1])
