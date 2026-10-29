#!/usr/bin/env python3
"""TRAD-style aligned step records for the raw-state blocks of an AgentRunbook-R reader prompt.

TRAD (Zhou et al., 2024, arXiv 2403.06221, §3.2-3.3) gives the decision model each retrieved step
with its neighbours, marked [Step -1 / 0 / +1] relative to the retrieved step. Here the retrieval is
unchanged (B1 prompt rows); only the presentation of each raw-state block changes:

  1. the states of a block are marked [Step -1] / [Step 0 | retrieved] / [Step +1] around the block's
     center state;
  2. Step 0 is shown in full except "site chrome": runs of >= MIN_RUN consecutive AXTree lines whose
     bid-free text occurs on >= CHROME_DF of the distinct states of the domain (navigation bars, menus,
     skip links). Isolated common lines (a "Number" label inside a form) stay in place. The removed
     lines are listed once, under "Common page elements", at the top of the raw-state section;
  3. a neighbour state shows only its lines that are not on Step 0 (what the action changed), plus
     the count of lines it shares with Step 0;
  4. a full action sequence repeated for the same trajectory, and a Step 0 state already shown in
     full in an earlier block, are replaced by a pointer to the earlier block.

Every line of the original memory context is still reachable (chrome once, shared lines via Step 0),
so the transform shortens the prompt without removing information other than the bids of chrome
lines and the position of chrome lines inside each page. Header lines (query label, rank, similarity,
trajectory, goal) are kept byte-for-byte, so this differs from the base prompt only in layout.

Controls, same chrome set, original layout otherwise: chrome_messages = step 2 only; dedup_messages = steps 2
and 4 (exact repeats -> pointers). trad - dedup isolates the step-aligned diff layout (steps 1 and 3).

  python e1/oneoff/trad_steps.py measure --src <per_question.jsonl or prompt_rows.jsonl> [...] [--layout trad|dedup|chrome]
      prints prompt length before/after and gold-phrase retention (local check, no model).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

CHROME_DF = 0.3
MIN_RUN = 5  # only runs of >= MIN_RUN consecutive chrome lines are moved out; isolated ones stay in place
BID_RE = re.compile(r"\[[A-Za-z]*\d+\]\s*")
STATE_RE = re.compile(r"(?m)^State (\d+) \(step (-?\d+)\)\n")
RAW_HEAD = "### Raw state result "


def norm(line: str) -> str:
    return BID_RE.sub("", line).strip()


def split_block(text: str):
    """-> dict(head, full, local, states=[(idx, step, url_action_lines, ax_lines)]) or None."""
    try:
        head, rest = text.split("\nFull action sequence\n", 1)
        full, rest = rest.split("\n\nLocal slice action sequence\n", 1)
        local, rest = rest.split("\n\nRelevant state slices\n", 1)
    except ValueError:
        return None
    m = re.search(r"(?m)^- Trajectory: (\S+)", head)
    c = re.search(r"(?m)^- Center state index: (\d+)", head)
    if not m or not c:
        return None
    states = []
    marks = list(STATE_RE.finditer(rest))
    for i, sm in enumerate(marks):
        body = rest[sm.end(): marks[i + 1].start() if i + 1 < len(marks) else len(rest)].rstrip("\n")
        pre, sep, ax = body.partition("- AXTree:\n")
        if not sep:
            return None
        states.append((int(sm.group(1)), sm.group(2), pre.rstrip("\n").split("\n"), ax.split("\n")))
    return {"head": head, "full": full, "local": local, "traj": m.group(1), "center": int(c.group(1)),
            "states": states}


def raw_blocks(messages):
    for m in messages:
        if isinstance(m.get("content"), list):
            for p in m["content"]:
                if p.get("type") == "text" and p["text"].startswith(RAW_HEAD):
                    yield p["text"]


def state_lines(messages):
    """{(trajectory, state index): set of normalized AXTree lines} over one prompt."""
    out = {}
    for t in raw_blocks(messages):
        b = split_block(t)
        if b:
            for idx, _, _, ax in b["states"]:
                out[(b["traj"], idx)] = {norm(x) for x in ax if norm(x)}
    return out


def build_chrome(prompts, df=CHROME_DF) -> set[str]:
    """Normalized AXTree lines on >= df of the distinct (trajectory, state) pages across `prompts`."""
    pages = {}
    for msgs in prompts:
        pages.update(state_lines(msgs))
    cnt = Counter(x for lines in pages.values() for x in lines)
    return {x for x, k in cnt.items() if k >= df * max(1, len(pages))}


def strip_chrome(ax: list[str], chrome: set[str], used_chrome: dict) -> list[str]:
    """Drop runs of >= MIN_RUN consecutive chrome lines (blank lines inside a run count as chrome)."""
    flag = [(not norm(x)) or norm(x) in chrome for x in ax]
    keep, i = [], 0
    while i < len(ax):
        j = i
        while j < len(ax) and flag[j]:
            j += 1
        if j - i >= MIN_RUN:
            for x in ax[i:j]:
                if norm(x):
                    used_chrome.setdefault(norm(x), None)
            i = j
            continue
        keep.extend(ax[i:max(j, i + 1)])
        i = max(j, i + 1)
    return keep


def render(text: str, k: int, chrome: set[str], seen_full: dict, seen_state: dict, used_chrome: dict) -> str:
    b = split_block(text)
    if b is None:
        return text
    traj, center = b["traj"], b["center"]
    out = [b["head"], "", "Full action sequence"]
    if traj in seen_full:
        out.append(f"(same trajectory {traj} as Raw state result {seen_full[traj]} above; see its full action sequence)")
    else:
        seen_full[traj] = k
        out.append(b["full"])
    out += ["", "Local slice action sequence", b["local"], "",
            "Aligned steps (Step 0 = the retrieved step; common page elements are listed once at the top of this section)"]
    by_idx = {s[0]: s for s in b["states"]}
    zero = by_idx.get(center)
    zero_set = {norm(x) for x in zero[3]} if zero else set()
    for idx, step, meta, ax in b["states"]:
        rel = idx - center
        tag = "[Step 0 | retrieved]" if rel == 0 else f"[Step {rel:+d}]"
        out.append(f"{tag} State {idx} (step {step})")
        out += meta
        keep = strip_chrome(ax, chrome, used_chrome)
        if rel == 0:
            if (traj, idx) in seen_state:
                out.append(f"- Page: identical to State {idx} shown in Raw state result {seen_state[(traj, idx)]} above")
            else:
                seen_state[(traj, idx)] = k
                out += ["- Page (common page elements omitted):"] + keep
        else:
            new = [x for x in keep if norm(x) not in zero_set]
            same = sum(1 for x in keep if norm(x) and norm(x) in zero_set)
            out.append(f"- Page vs Step 0: {same} lines identical to Step 0 omitted; lines only on this page:")
            out += new if new else ["(none)"]
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def trad_messages(messages, chrome: set[str]):
    """Copy of messages with every raw-state block re-laid-out; returns (messages, n_blocks)."""
    seen_full, seen_state = {}, {}
    return _transform(messages, lambda t, k, used: render(t, k, chrome, seen_full, seen_state, used))


def chrome_only(text: str, chrome: set[str], used_chrome: dict, k: int = 0, seen: dict | None = None) -> str:
    """The block in its original layout, with the chrome runs removed from each state's AXTree. With
    `seen` (dedup control) also: a full action sequence or a state already shown earlier in the prompt is
    replaced by a pointer. No step marks, no diffs."""
    b = split_block(text)
    if b is None:
        return text
    head, rest = text.split("\n\nRelevant state slices\n", 1)
    if seen is not None:
        pre_full, full_rest = head.split("\nFull action sequence\n", 1)
        full, local_rest = full_rest.split("\n\nLocal slice action sequence\n", 1)
        if ("full", b["traj"]) in seen:
            full = f"(same trajectory {b['traj']} as Raw state result {seen[('full', b['traj'])]} above; see its full action sequence)"
        else:
            seen[("full", b["traj"])] = k
        head = pre_full + "\nFull action sequence\n" + full + "\n\nLocal slice action sequence\n" + local_rest
    marks = list(STATE_RE.finditer(rest))
    out = [head, "\n\nRelevant state slices\n", rest[: marks[0].start()] if marks else rest]
    for i, sm in enumerate(marks):
        body = rest[sm.start(): marks[i + 1].start() if i + 1 < len(marks) else len(rest)]
        pre, sep, ax = body.partition("- AXTree:\n")
        core = ax.rstrip("\n")
        key = (b["traj"], int(sm.group(1)))
        if seen is not None and key in seen:
            lines = [f"(identical to State {key[1]} shown in Raw state result {seen[key]} above)"]
        else:
            if seen is not None:
                seen[key] = k
            lines = strip_chrome(core.split("\n"), chrome, used_chrome)
        out.append(pre + sep + "\n".join(lines) + ax[len(core):])
    return "".join(out)


def dedup_messages(messages, chrome: set[str]):
    """Control for trad_messages: steps 2 and 4 only (chrome listed once; an exactly repeated action
    sequence or state replaced by a pointer) in the original layout; neighbours stay in full, no step marks."""
    seen = {}
    return _transform(messages, lambda t, k, used: chrome_only(t, chrome, used, k, seen))


def chrome_messages(messages, chrome: set[str]):
    """Control for trad_messages: only the chrome step (2) -- chrome runs listed once at the top of the
    raw-state section and dropped from every state; no step marks, no diffs, no pointers."""
    return _transform(messages, lambda t, k, used: chrome_only(t, chrome, used))


def _transform(messages, fn):
    used_chrome, n = {}, 0
    out, title_ref = [], None
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if p.get("type") == "text" and p["text"].startswith(RAW_HEAD):
                    n += 1
                    p = {**p, "text": fn(p["text"], n, used_chrome)}
                elif p.get("type") == "text" and p["text"].startswith("## Retrieved Raw State Slices"):
                    p = dict(p)
                    title_ref = p
                parts.append(p)
            m = {**m, "content": parts}
        out.append(m)
    if title_ref is not None and used_chrome:
        title_ref["text"] = (title_ref["text"].rstrip("\n") + "\n\n### Common page elements (on most pages of this "
                             "site; shown once here and omitted from every state below)\n"
                             + "\n".join(used_chrome) + "\n\n")
    return out, n


LAYOUTS = {"trad": trad_messages, "dedup": dedup_messages, "chrome": chrome_messages}


def text_of(messages) -> str:
    return "\n".join(p["text"] for m in messages if isinstance(m.get("content"), list)
                     for p in m["content"] if p.get("type") == "text")


def measure(args) -> None:
    rows = []
    for path in args.src:
        for line in open(path):
            r = json.loads(line)
            if r.get("is_abstention_problem") or r["eval_function"].startswith("llm_"):
                continue
            msgs = r.get("messages") or r.get("prompt_messages")
            rows.append((r["question_id"], r["answer_gold"], msgs, path))
    by_src = {}
    for q, g, msgs, path in rows:
        by_src.setdefault(path, []).append(msgs)
    chrome = {p: build_chrome(v, args.df) for p, v in by_src.items()}
    for p, c in chrome.items():
        print(f"[measure] {p}: {len(by_src[p])} prompts, {len(c)} chrome lines (df >= {args.df})")
    before = after = 0
    lost, ratios = [], []
    for q, gold, msgs, path in rows:
        t0 = text_of(msgs)
        new, _ = LAYOUTS[args.layout](msgs, chrome[path])
        t1 = text_of(new)
        before += len(t0); after += len(t1); ratios.append(len(t1) / max(1, len(t0)))
        for ph in [x.strip().lower() for x in re.split(r"[,;]", gold) if x.strip()]:
            if ph in t0.lower() and ph not in t1.lower():
                lost.append((q, ph))
    ratios.sort()
    print(f"[measure] {len(rows)} prompts: chars {before:,} -> {after:,} ({after / before:.3f}); "
          f"per-prompt ratio median {ratios[len(ratios) // 2]:.3f}, p10 {ratios[len(ratios) // 10]:.3f}, "
          f"p90 {ratios[9 * len(ratios) // 10]:.3f}")
    print(f"[measure] gold phrases present before but not after: {len(lost)} {lost[:10]}")
    if args.show:
        for q, gold, msgs, path in rows:
            if q == args.show:
                print(text_of(LAYOUTS[args.layout](msgs, chrome[path])[0])[: args.chars])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["measure"])
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--df", type=float, default=CHROME_DF)
    ap.add_argument("--layout", choices=list(LAYOUTS), default="trad")
    ap.add_argument("--show", default=None)
    ap.add_argument("--chars", type=int, default=6000)
    measure(ap.parse_args())


if __name__ == "__main__":
    main()
