#!/usr/bin/env python3
"""Where are the answer phrases of the C2 stable fixes (RR vs RR+C), and what did the reader answer?

For each question: the answer phrases (official split; gold string if not phrase-scored), their hits per
context item in the RR pre-truncation context, the RR prompt row (official truncation) and the RR+C row
(within-block truncation): item index, kind, state index, char offset in the state AXTree and whether the
hit is past the C cut. Plus the 4 reader samples (c2rr_s1/2, c2rc_s1/2).

  python -I e1/oneoff/c2_fix_positions.py 44c05e59 597ca614 ...  -> ~/e1/v2work/reports/c2_fix_positions.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_keys  # noqa: E402
from trunc_c import is_raw, parse_block  # noqa: E402
from e1.common.io import iter_jsonl  # noqa: E402

E1 = step_keys.E1
CUT = re.compile(r"\n\.\.\.\[truncated (\d+) chars\]\.\.\.\n$")


def items(mc, norm):
    """Per text item: kind, normalised text, and per raw state (header, lower AXTree, normalised AXTree)."""
    out = []
    for i, x in enumerate(mc):
        if x.get("type") != "text":
            out.append({"i": i, "kind": x.get("type"), "norm": "", "states": []})
            continue
        v = x["value"]
        kind = "raw" if is_raw(x) else v.split("\n", 1)[0][:40]
        st = []
        if is_raw(x):
            try:
                _, states, _ = parse_block(v)
            except Exception:  # the officially truncated last item may not parse
                states = []
            st = [(h.split("\n", 1)[0], ax) for h, ax in states]
        out.append({"i": i, "kind": kind, "norm": " " + norm(v) + " ", "states": st, "len": len(v)})
    return out


def hits(its, phrases, norm):
    res = {}
    for p in phrases:
        hs = []
        for it in its:
            if f" {p} " not in it["norm"]:
                continue
            h = {"item": it["i"], "kind": it["kind"]}
            if it["states"]:
                sk = []
                for k, (hdr, ax) in enumerate(it["states"]):
                    if f" {p} " in " " + norm(ax) + " ":
                        m = CUT.search(ax)
                        body = ax[: m.start()] if m else ax
                        off = body.lower().find(p.split()[0]) if p.split() else -1
                        sk.append({"state": k, "hdr": hdr, "ax_len": len(body), "cut": int(m.group(1)) if m else 0,
                                   "first_word_off": off})
                h["states"] = sk
            hs.append(h)
        res[p] = hs
    return res


def main() -> None:
    from e1.analysis.retrieval_eval import _official_metrics, answer_phrases
    from e1.pipeline.retrieve import meta
    from reader_interventions import correct, correct_strict
    norm = _official_metrics()[0]
    want = sys.argv[1:]
    report = {}
    for dom in step_keys.DOMS:
        M = meta(dom)
        qs = {q for q in M if any(q.startswith(w) for w in want)}
        if not qs:
            continue
        pre = {r["question_id"]: r["memory_context"] for r in iter_jsonl(E1 / f"e2work/contexts/rr/{dom}_contexts.jsonl") if r["question_id"] in qs}
        rr = {r["question_id"]: r for r in iter_jsonl(E1 / f"runs/rr_small/{dom}_small/prompt_rows.jsonl") if r["question_id"] in qs}
        rc = {r["question_id"]: r for r in iter_jsonl(E1 / f"runs/c2_rc_small/{dom}_small/prompt_rows.jsonl") if r["question_id"] in qs}
        outs = {}
        for v in ["c2rr_s1", "c2rr_s2", "c2rc_s1", "c2rc_s2"]:
            for o in iter_jsonl(E1 / f"runs/iv_{v}_small/{dom}_small/outputs.jsonl"):
                if o["question_id"] in qs:
                    ok = bool(correct(o["wrap2_boxed"], o)) if o["cap_no_answer"] else correct_strict(o["content"], o)
                    outs.setdefault(o["question_id"], {})[v] = {
                        "ok": ok, "cap": o["cap_no_answer"], "boxed": o.get("wrap2_boxed") if o["cap_no_answer"] else o.get("boxed"),
                        "reasoning_chars": len(o.get("reasoning") or ""), "reasoning_tail": (o.get("reasoning") or "")[-600:]}
        for q in sorted(qs):
            m = M[q]
            ph = answer_phrases(m)
            phrases = ph if ph else [norm(m["answer_gold"])]
            ctx = {"pre": pre[q], "rr": rr[q]["memory_context"], "rc": rc[q]["memory_context"]}
            I = {k: items(v, norm) for k, v in ctx.items()}
            report[q] = {
                "dom": dom, "question": m["question_text"], "gold": m["answer_gold"], "eval": m["eval_function"],
                "phrase_scored": ph is not None, "phrases": phrases,
                "n_items": {k: len(v) for k, v in ctx.items()},
                "n_raw": {k: sum(map(is_raw, v)) for k, v in ctx.items()},
                "rr_tokens": [rr[q].get("memory_context_original_token_count"), rr[q].get("memory_context_was_truncated")],
                "hits": {k: hits(v, phrases, norm) for k, v in I.items()},
                "item_kinds_pre": [it["kind"] for it in I["pre"]],
                "reader": outs.get(q, {}),
            }
    out = step_keys.V2 / "reports/c2_fix_positions.json"
    out.write_text(json.dumps(report, indent=1))
    for q, r in report.items():
        print(f"\n=== {q} [{r['dom']}] phrase_scored={r['phrase_scored']} items pre/rr/rc={r['n_items']} raw={r['n_raw']}")
        print("Q:", r["question"][:300]); print("gold:", r["gold"][:200])
        for k in ("pre", "rr", "rc"):
            for p, hs in r["hits"][k].items():
                print(f"  {k:3} {p[:40]!r}: " + "; ".join(
                    f"it{h['item']}({h['kind'][:12]})" + ("".join(f"[s{s['state']} off{s['first_word_off']}/{s['ax_len']} cut{s['cut']}]" for s in h.get("states", [])))
                    for h in hs) if hs else f"  {k:3} {p[:40]!r}: -")
        for v, o in sorted(r["reader"].items()):
            print(f"  {v}: ok={o['ok']} cap={o['cap']} boxed={str(o['boxed'])[:120]!r} reas={o['reasoning_chars']}")


if __name__ == "__main__":
    main()
