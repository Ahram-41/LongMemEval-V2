#!/usr/bin/env python3
"""Where did the point die? Per-question first failing stage on phrase-answer normal questions
(answer = all gold phrases, official split / normalisation; presence = word-bounded match after
the official normalize_phrase; a proxy: a phrase can occur in an unrelated context).

  1 not_in_store   : the phrases are not all present in the union of the four pools' text
                     (raw: goal, action sequences, URLs, slice text; events: action + description;
                     notes: title, description, content)
  2 invisible      : present, but not all phrases are present in the union of what the indexes
                     can see: raw entries' embedded prefix (official 40k-char middle cut, first
                     4,096 tokens) + events + notes (short texts, treated as fully embedded)
  3 not_retrieved  : visible, but not all phrases in the retrieved memory context before the 200k
                     truncation (notes + events + raw blocks as rendered, query lines removed)
  4 cut_at_assembly: retrieved, but not all phrases in the context the reader actually received
  5 read_wrong     : in the reader's context, answer judged wrong
  6 correct        : in the reader's context, answer judged correct
For B1 the pre-truncation context exists only for untruncated questions; for the 107 truncated
ones stages 3 and 4 cannot be separated and are reported as "3or4". For an E2 arm the
pre-truncation contexts are build_context's output, so all six stages are defined.
A question that fails at stage 1-4 can still be judged correct (proxy noise or prior knowledge);
the count is reported per stage.
Usage: python -I e1/analysis/waterfall.py --arm b1_small
       python -I e1/analysis/waterfall.py --arm e2_fusion_merged_small --contexts ~/e1/e2work/contexts/e2_fusion
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.analysis.retrieval_eval import _official_metrics, answer_phrases  # noqa: E402
from e1.common.io import iter_jsonl  # noqa: E402

E1 = Path("~/e1").expanduser()
QUERY_LINE = re.compile(r"^- (Note query|Event query|Source raw query): .*$", re.M)
STAGES = ["1_not_in_store", "2_invisible", "3_not_retrieved", "3or4_not_in_reader_context", "4_cut_at_assembly",
          "5_read_wrong", "6_correct"]
TYPES = ["static-environment", "dynamic-environment", "procedure", "errors-gotchas"]


def ctx_text(ctx: list[dict], norm) -> str:
    return " " + norm(QUERY_LINE.sub(" ", "\n".join(x["value"] for x in ctx if x.get("type") == "text"))) + " "


def has_all(ph, text):
    return all(f" {p} " in text for p in ph)


def pool_texts(dom: str, norm):
    d = E1 / f"index/agentrunbook_r_{dom}_small/memory_state/pools"
    ET = pickle.loads((E1 / "e2work/strata" / f"{dom}_entrytext.pkl").read_bytes())
    raw_full, raw_meta = [], []
    for e in iter_jsonl(d / "raw_state_pool.jsonl"):
        raw_meta.append(norm("\n".join([e["goal"], e["full_action_sequence"], e["slice_action_sequence"], "\n".join(e["slice_urls"])])))
        del e
    raw_full = [m + " " + t for m, t in zip(raw_meta, ET["full_norm"])]
    events = [norm(e["local_action"] + "\n" + e["description"]) for e in iter_jsonl(d / "event_pool.jsonl")]
    notes = [norm("\n".join([e["title"], e["description"], e["content"]]))
             for f in ["procedure_note_pool.jsonl", "hint_note_pool.jsonl"] for e in iter_jsonl(d / f)]
    other = " | ".join(events + notes)
    store = " " + " | ".join(raw_full) + " | " + other + " "
    visible = " " + " | ".join(ET["emb_norm"]) + " | " + other + " "
    return store, visible


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="b1_small")
    ap.add_argument("--contexts", default=None, help="dir with <dom>_contexts.jsonl (pre-truncation) for an E2 arm")
    ap.add_argument("--out", default=str(E1 / "e2work/rounds/waterfall"))
    a = ap.parse_args()
    norm = _official_metrics()[0]
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    tab = defaultdict(Counter); cor = defaultdict(Counter)
    for dom in ["web", "enterprise"]:
        M = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/meta/{dom}.jsonl")}
        Q = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl")}
        elig = {q: answer_phrases(m) for q, m in M.items()}
        elig = {q: p for q, p in elig.items() if p}
        store, visible = pool_texts(dom, norm)
        pre = {}
        if a.contexts:
            for c in iter_jsonl(Path(a.contexts).expanduser() / f"{dom}_contexts.jsonl"):
                if c["question_id"] in elig:
                    pre[c["question_id"]] = ctx_text(c["memory_context"], norm)
        rows = []
        for r in iter_jsonl(E1 / f"runs/{a.arm}/{dom}_small/per_question.jsonl"):
            q = r["question_id"]
            if q not in elig:
                del r
                continue
            ph = elig[q]
            reader = ctx_text(r["memory_context"], norm)
            ok = bool(r["score_bool"])
            truncated = bool(r.get("memory_context_was_truncated"))
            if not has_all(ph, store):
                st = "1_not_in_store"
            elif not has_all(ph, visible):
                st = "2_invisible"
            else:
                if a.contexts:
                    before = pre[q]
                elif not truncated:
                    before = reader
                else:
                    before = None
                if before is not None and not has_all(ph, before):
                    st = "3_not_retrieved"
                elif not has_all(ph, reader):
                    st = "4_cut_at_assembly" if before is not None else "3or4_not_in_reader_context"
                else:
                    st = "6_correct" if ok else "5_read_wrong"
            rows.append({"question_id": q, "domain": dom, "question_type": M[q]["question_type"], "stage": st,
                         "correct": ok, "phrases": ph, "reader_context_truncated": truncated,
                         "query_source": "regenerated" if (Q[q]["context_truncated"] or Q[q]["missing_detected"]) else "parsed",
                         "phrase_in_store": [f" {p} " in store for p in ph], "phrase_visible": [f" {p} " in visible for p in ph],
                         "phrase_in_reader_context": [f" {p} " in reader for p in ph]})
            tab[(dom, M[q]["question_type"])][st] += 1; cor[(dom, M[q]["question_type"])][st] += ok
            del r
        with open(out / f"{a.arm}_{dom}.jsonl", "w") as fp:
            for x in rows:
                fp.write(json.dumps(x) + "\n")
    print(f"waterfall for {a.arm}: first failing stage, count (of which judged correct)")
    print(f"{'domain':11s} {'type':22s} {'n':>4s} " + " ".join(f"{s[:14]:>15s}" for s in STAGES))
    for dom in ["web", "enterprise", "ALL"]:
        for t in TYPES + ["ALL"]:
            c, k = Counter(), Counter()
            for (d, tt), v in tab.items():
                if (dom == "ALL" or d == dom) and (t == "ALL" or tt == t):
                    c.update(v); k.update(cor[(d, tt)])
            n = sum(c.values())
            if n:
                print(f"{dom:11s} {t:22s} {n:4d} " + " ".join(f"{c[s]:>8d} ({k[s]:>3d})" for s in STAGES))


if __name__ == "__main__":
    main()
