#!/usr/bin/env python3
"""E3 inputs (CPU only). Run with the vllm env python (it has `tokenizers` and numpy), in a CPU job:
the dataset is one JSON array of a few hundred MB.

  schema : verify the schema at source -> ~/e1/e3work/prep/schema.json (+ printed summary)
  build  : derived, compact inputs under ~/e1/e3work/prep/
             questions.jsonl  one row per question: id, type, abstention flag, question, answer, date,
                              haystack session ids / dates, answer_session_ids, and the flat round list of
                              the haystack (r_uid = row in rounds.jsonl, r_sess = session position,
                              r_ev = 1 if the round holds a has_answer turn in THIS question's copy)
             rounds.jsonl     unique round texts (dedup by sha1 of the text), with token count
             pieces.jsonl     rounds cut into consecutive <= 4,096-token pieces (Qwen3-Embedding-8B
                              tokenizer, no special tokens, cut at token offsets = exact substrings)

A round = one user turn plus the assistant turn(s) that follow it up to the next user turn; assistant
turns before the first user turn form a round of their own. Round text: "user: ...\\nassistant: ...".
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.manifest import write_manifest  # noqa: E402

E1 = Path("~/e1").expanduser()
DATA = E1 / "data/longmemeval-s/longmemeval_s_cleaned.json"
WORK = E1 / "e3work"
PREP = WORK / "prep"
TOKENIZER_GLOB = "hf_cache/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/*/tokenizer.json"
MAX_TOKENS = 4096


def tokenizer():
    from tokenizers import Tokenizer
    paths = sorted(E1.glob(TOKENIZER_GLOB))
    assert len(paths) == 1, paths
    return Tokenizer.from_file(str(paths[0])), paths[0]


def load() -> list[dict]:
    with open(DATA) as fp:
        return json.load(fp)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def session_rounds(turns: list[dict]) -> list[list[int]]:
    """Turn indices of each round (see module doc)."""
    rounds: list[list[int]] = []
    for i, t in enumerate(turns):
        if t.get("role") == "user" or not rounds:
            rounds.append([i])
        else:
            rounds[-1].append(i)
    return rounds


def round_text(turns: list[dict], idx: list[int]) -> str:
    return "\n".join(f"{turns[i].get('role', '?')}: {turns[i].get('content', '')}" for i in idx)


def dist(xs) -> dict:
    xs = list(xs)
    if not xs:
        return {}
    return {"n": len(xs), "min": min(xs), "median": statistics.median(xs), "mean": round(statistics.mean(xs), 2),
            "max": max(xs), "sum": sum(xs)}


def do_schema() -> None:
    data = load()
    tok, tok_path = tokenizer()
    out: dict = {"file": str(DATA), "n_questions": len(data), "top_type": type(data).__name__}
    keysets = collections.Counter(tuple(sorted(q.keys())) for q in data)
    out["question_keysets"] = {"|".join(k): v for k, v in keysets.items()}
    out["question_type"] = dict(collections.Counter(q["question_type"] for q in data))
    out["abstention_by_type"] = dict(collections.Counter(q["question_type"] for q in data if q["question_id"].endswith("_abs")))
    out["n_abs"] = sum(q["question_id"].endswith("_abs") for q in data)
    out["question_id_unique"] = len({q["question_id"] for q in data}) == len(data)
    turn_keys = collections.Counter()
    roles = collections.Counter()
    has_answer_vals = collections.Counter()
    sess_per_q, ans_per_q = [], collections.defaultdict(list)
    lens_match = ans_subset = 0
    q_with_ha, ha_outside_answer = 0, 0
    ha_turns_per_q = collections.defaultdict(list)
    abs_ans = []
    sess_content: dict[str, str] = {}
    sess_conflict = sess_conflict_text_only = 0
    sess_occ = collections.Counter()
    date_types = collections.Counter()
    answer_types = collections.Counter()
    for q in data:
        hs, hd, hss, ans = q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"], q["answer_session_ids"]
        sess_per_q.append(len(hs))
        lens_match += len(hs) == len(hd) == len(hss)
        ans_subset += set(ans) <= set(hs)
        ans_per_q[q["question_type"]].append(len(ans))
        answer_types[type(q.get("answer")).__name__] += 1
        date_types[type(hd[0]).__name__ if hd else "none"] += 1
        if q["question_id"].endswith("_abs"):
            abs_ans.append(len(ans))
        n_ha = 0
        for sid, turns in zip(hs, hss):
            sess_occ[sid] += 1
            for t in turns:
                turn_keys[tuple(sorted(t.keys()))] += 1
                roles[t.get("role")] += 1
                if "has_answer" in t:
                    has_answer_vals[repr(t["has_answer"])] += 1
                    if t["has_answer"]:
                        n_ha += 1
                        if sid not in ans:
                            ha_outside_answer += 1
            full = json.dumps(turns, sort_keys=True)
            txt = json.dumps([[t.get("role"), t.get("content")] for t in turns])
            if sid in sess_content:
                if sess_content[sid][0] != full:
                    sess_conflict += 1
                    if sess_content[sid][1] != txt:
                        sess_conflict_text_only += 1
            else:
                sess_content[sid] = (full, txt)
        q_with_ha += n_ha > 0
        ha_turns_per_q[q["question_type"] + ("_abs" if q["question_id"].endswith("_abs") else "")].append(n_ha)
    out["turn_keysets"] = {"|".join(k): v for k, v in turn_keys.items()}
    out["roles"] = dict(roles)
    out["has_answer_values"] = dict(has_answer_vals)
    out["sessions_per_question"] = dist(sess_per_q)
    out["lengths_match_ids_dates_sessions"] = lens_match
    out["answer_ids_subset_of_haystack"] = ans_subset
    out["answer_sessions_per_question_by_type"] = {k: dict(collections.Counter(v)) for k, v in ans_per_q.items()}
    out["abs_answer_sessions"] = dict(collections.Counter(abs_ans))
    out["questions_with_has_answer_turn"] = q_with_ha
    out["has_answer_turns_outside_answer_sessions"] = ha_outside_answer
    out["has_answer_turns_per_question_by_type"] = {k: dist(v) for k, v in ha_turns_per_q.items()}
    out["answer_value_types"] = dict(answer_types)
    out["haystack_date_types"] = dict(date_types)
    out["example_dates"] = data[0]["haystack_dates"][:2]
    out["example_question_date"] = data[0].get("question_date")
    out["session_occurrences_total"] = sum(sess_occ.values())
    out["unique_session_ids"] = len(sess_occ)
    out["session_id_reused_with_different_content"] = sess_conflict
    out["  of_which_text_differs"] = sess_conflict_text_only
    out["sessions_occurring_in_n_questions"] = dist(sess_occ.values())
    # rounds and tokens over unique sessions (by session id; first occurrence) and unique round texts
    n_rounds_unique_sess, tok_unique_sess = 0, 0
    round_tok: dict[str, int] = {}
    seen = set()
    for q in data:
        for sid, turns in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            if sid in seen:
                continue
            seen.add(sid)
            for idx in session_rounds(turns):
                n_rounds_unique_sess += 1
                t = round_text(turns, idx)
                h = sha1(t)
                if h not in round_tok:
                    round_tok[h] = len(tok.encode(t, add_special_tokens=False).ids)
                tok_unique_sess += round_tok[h]
    out["tokenizer"] = str(tok_path)
    out["rounds_in_unique_sessions"] = n_rounds_unique_sess
    out["tokens_in_unique_sessions"] = tok_unique_sess
    out["unique_round_texts"] = len(round_tok)
    out["tokens_unique_round_texts"] = sum(round_tok.values())
    out["round_tokens"] = dist(round_tok.values())
    out["rounds_over_4096_tokens"] = sum(v > MAX_TOKENS for v in round_tok.values())
    out["pieces_needed"] = sum(max(1, -(-v // MAX_TOKENS)) for v in round_tok.values())
    # haystack tokens per question (with duplicates across questions), for scale
    per_q = []
    for q in data:
        n = 0
        for turns in q["haystack_sessions"]:
            for idx in session_rounds(turns):
                n += round_tok[sha1(round_text(turns, idx))]
        per_q.append(n)
    out["haystack_round_tokens_per_question"] = dist(per_q)
    out["question_lengths_chars"] = dist(len(q["question"]) for q in data)
    PREP.mkdir(parents=True, exist_ok=True)
    (PREP / "schema.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    write_manifest(PREP, inputs={"data": DATA, "tokenizer": tok_path}, params={"what": "schema"}, name="manifest_schema.json")
    print(json.dumps(out, indent=2, default=str))


def do_build() -> None:
    data = load()
    tok, tok_path = tokenizer()
    PREP.mkdir(parents=True, exist_ok=True)
    uid_of: dict[str, int] = {}
    rounds_fp = open(PREP / "rounds.jsonl", "w")
    pieces_fp = open(PREP / "pieces.jsonl", "w")
    n_pieces = 0
    with open(PREP / "questions.jsonl", "w") as qfp:
        for q in data:
            r_uid, r_sess, r_ev, r_turns = [], [], [], []
            ev_turns = []
            for si, (sid, turns) in enumerate(zip(q["haystack_session_ids"], q["haystack_sessions"])):
                for ti, t in enumerate(turns):
                    if t.get("has_answer"):
                        ev_turns.append([si, ti])
                for idx in session_rounds(turns):
                    text = round_text(turns, idx)
                    h = sha1(text)
                    if h not in uid_of:
                        uid = uid_of[h] = len(uid_of)
                        enc = tok.encode(text, add_special_tokens=False)
                        offs, n = enc.offsets, len(enc.ids)
                        rounds_fp.write(json.dumps({"uid": uid, "sha1": h, "n_tokens": n, "text": text}, ensure_ascii=True) + "\n")
                        for k, a in enumerate(range(0, max(n, 1), MAX_TOKENS)):
                            b = min(n, a + MAX_TOKENS)
                            if n == 0:
                                c0, c1 = 0, len(text)
                            else:
                                c0 = 0 if k == 0 else offs[a][0]
                                c1 = len(text) if b == n else offs[b][0]
                            pieces_fp.write(json.dumps({"piece_id": n_pieces, "uid": uid, "piece": k, "n_tokens": b - a,
                                                        "text": text[c0:c1]}, ensure_ascii=True) + "\n")
                            n_pieces += 1
                    r_uid.append(uid_of[h]); r_sess.append(si); r_turns.append(idx)
                    r_ev.append(int(any(turns[i].get("has_answer") for i in idx)))
            row = {k: q[k] for k in ("question_id", "question_type", "question", "answer", "question_date",
                                     "haystack_session_ids", "haystack_dates", "answer_session_ids")}
            row["is_abs"] = q["question_id"].endswith("_abs")
            row.update(r_uid=r_uid, r_sess=r_sess, r_ev=r_ev, r_turns=r_turns, ev_turns=ev_turns)
            qfp.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
    rounds_fp.close(); pieces_fp.close()
    print(f"build: {len(data)} questions, {len(uid_of)} unique rounds, {n_pieces} pieces -> {PREP}")
    write_manifest(PREP, inputs={"data": DATA, "tokenizer": tok_path}, params={"what": "build", "max_tokens": MAX_TOKENS},
                   name="manifest_build.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["schema", "build"])
    a = ap.parse_args()
    do_schema() if a.what == "schema" else do_build()


if __name__ == "__main__":
    main()
