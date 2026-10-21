#!/usr/bin/env python3
"""E2 §十 S1: memory form x question type with the four existing AgentRunbook-R pools.

S1a retention : is the answer stored at all? For each normal question with a phrase answer
                (norm_phrase_set_match*, official split / normalisation), check whether every
                answer phrase occurs (word-bounded) somewhere in the WHOLE text of each pool,
                no retrieval. All questions of a domain share one haystack (100 trajectories).
                Pool text = what the renderer shows from each entry:
                  raw   : goal, action sequences, URLs, state accessibility trees
                  event : local action + description
                  procedure / hint note : title, description, content
S1b retrieved : in B1's actual memory context (as the reader saw it, after 200k truncation),
                in which section do the answer phrases occur (notes / events / raw), with the
                query header lines removed so a query cannot cover its own answer.
Both are reported per question_type x domain; questions are only the phrase-answer normal
ones (static / dynamic / procedure); errors-gotchas and multiple-choice have no phrase answer and
false-premise questions have no answer phrase, so they are counted as not eligible.
Usage: python -I e1/analysis/memory_form.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.analysis.retrieval_eval import _official_metrics, answer_phrases  # noqa: E402
from e1.common.io import iter_jsonl, load_rows  # noqa: E402

E1 = Path("~/e1").expanduser()
POOLS = ["raw", "event", "procedure", "hint"]
TYPES = ["static-environment", "dynamic-environment", "procedure", "errors-gotchas"]
NOTE_T = "## Procedure and Hint Notes Learned from Previous Tasks in the Environment\n\n"
EVENT_T = "## Retrieved State-Transition Events from Previous Tasks in the Environment\n\n"
RAW_T = "## Retrieved Raw State Slices from Previous Tasks in the Environment\n\n"
QUERY_LINE = re.compile(r"^- (Note query|Event query|Source raw query): .*$", re.M)


def pool_texts(dom: str) -> dict[str, str]:
    norm = _official_metrics()[0]
    d = E1 / f"index/agentrunbook_r_{dom}_small/memory_state/pools"
    out = {}
    parts = []
    with open(d / "raw_state_pool.jsonl") as fp:
        for line in fp:
            e = json.loads(line)
            parts.append(norm("\n".join([e["goal"], e["full_action_sequence"], e["slice_action_sequence"],
                                         "\n".join(e["slice_urls"]), e["slice_axtree_text"]])))
            del e
    out["raw"] = " " + " | ".join(parts) + " "
    out["event"] = " " + " | ".join(norm(e["local_action"] + "\n" + e["description"]) for e in iter_jsonl(d / "event_pool.jsonl")) + " "
    for k, f in [("procedure", "procedure_note_pool.jsonl"), ("hint", "hint_note_pool.jsonl")]:
        out[k] = " " + " | ".join(norm("\n".join([e["title"], e["description"], e["content"]])) for e in iter_jsonl(d / f)) + " "
    return out


def sections(ctx: list[dict]) -> dict[str, str]:
    norm = _official_metrics()[0]
    cur, buf = None, defaultdict(list)
    for it in ctx:
        if it.get("type") != "text":
            continue
        v = it["value"]
        if v.startswith(NOTE_T):
            cur = "notes"
        elif v.startswith(EVENT_T):
            cur = "events"
        elif v.startswith(RAW_T):
            cur = "raw"
        if cur:
            buf[cur].append(QUERY_LINE.sub(" ", v))
    return {k: " " + norm("\n".join(v)) + " " for k, v in buf.items()}


def covered(ph: list[str], text: str) -> bool:
    return all(f" {p} " in text for p in ph)


def main() -> None:
    b1 = load_rows(E1 / "runs", "b1_small", ("question_id", "score_bool"))
    for dom in ["web", "enterprise"]:
        M = [r for r in iter_jsonl(E1 / f"cache/meta/{dom}.jsonl")]
        P = pool_texts(dom)
        print(f"\n=== {dom}: pool sizes (normalised chars): " + ", ".join(f"{k} {len(v):,}" for k, v in P.items()))
        # S1a
        print("S1a retention: questions whose answer phrases ALL occur in the whole pool (no retrieval)")
        print(f"   {'type':22s} {'eligible':>8s} " + " ".join(f"{k:>10s}" for k in POOLS) + f" {'notes(any)':>10s} {'only raw':>9s}")
        elig = {r["question_id"]: answer_phrases(r) for r in M}
        for t in TYPES + ["ALL"]:
            Q = [r for r in M if not r["is_abstention_problem"] and (t == "ALL" or r["question_type"] == t)]
            E = [r for r in Q if elig[r["question_id"]]]
            if not E:
                print(f"   {t:22s} {0:>4d}/{len(Q):<3d} (no phrase answers)")
                continue
            hit = {k: sum(covered(elig[r["question_id"]], P[k]) for r in E) for k in POOLS}
            notes_any = sum(covered(elig[r["question_id"]], P["procedure"]) or covered(elig[r["question_id"]], P["hint"]) for r in E)
            only_raw = sum(covered(elig[r["question_id"]], P["raw"]) and not any(covered(elig[r["question_id"]], P[k]) for k in ["event", "procedure", "hint"]) for r in E)
            print(f"   {t:22s} {len(E):>4d}/{len(Q):<3d} " + " ".join(f"{hit[k]:>4d} ({hit[k]/len(E):.2f})" for k in POOLS)
                  + f" {notes_any:>4d} ({notes_any/len(E):.2f}) {only_raw:>4d}")
        # S1b
        print("S1b B1 context (as the reader saw it): answer phrases ALL found in section; B1 accuracy by where found")
        print(f"   {'type':22s} {'eligible':>8s} {'notes':>11s} {'events':>11s} {'raw':>11s} {'none':>11s} {'raw only':>11s} | B1 correct: raw-only / not-raw-but-notes|events / none")
        sec = {}
        for r in iter_jsonl(E1 / f"runs/b1_small/{dom}_small/per_question.jsonl"):
            if elig.get(r["question_id"]):
                sec[r["question_id"]] = sections(r["memory_context"])
            del r
        for t in TYPES + ["ALL"]:
            E = [r["question_id"] for r in M if not r["is_abstention_problem"] and elig[r["question_id"]] and (t == "ALL" or r["question_type"] == t)]
            if not E:
                continue
            f = {k: [q for q in E if covered(elig[q], sec[q].get(k, ""))] for k in ["notes", "events", "raw"]}
            none = [q for q in E if not any(q in f[k] for k in f)]
            raw_only = [q for q in f["raw"] if q not in f["notes"] and q not in f["events"]]
            other = [q for q in E if q not in f["raw"] and (q in f["notes"] or q in f["events"])]
            acc = lambda S: f"{sum(bool(b1[q]['score_bool']) for q in S)}/{len(S)}"
            print(f"   {t:22s} {len(E):>8d} " + " ".join(f"{len(f[k]):>4d} ({len(f[k])/len(E):.2f})" for k in ["notes", "events", "raw"])
                  + f" {len(none):>4d} ({len(none)/len(E):.2f}) {len(raw_only):>4d} ({len(raw_only)/len(E):.2f}) | {acc(raw_only)} / {acc(other)} / {acc(none)}")


if __name__ == "__main__":
    main()
