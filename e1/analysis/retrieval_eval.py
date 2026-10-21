#!/usr/bin/env python3
"""E2 retrieval-only evaluation (no reader). Two checkable proxies, computed on the merged raw
blocks of each method (the blocks that would enter the context, before the 200k truncation):

  ans : answer-phrase coverage. Normal questions scored by norm_phrase_set_match(_ordered); the
        gold answer is split with the official split_phrases (the question's own separators and
        normalisation). Covered = every phrase occurs (word-bounded, after the official
        normalize_phrase) in the union of the retrieved blocks. Also reports the phrase share.
  anc : quoted-anchor coverage (E1 page_coverage logic). Questions naming quoted UI terms;
        covered = some single block contains all anchors ("full").
Block text = the entry's goal, action sequences, URLs and state accessibility trees (official pool
fields), lower-cased; unlike E1 page_coverage, the "Source raw query" header is excluded, so a
query that repeats a quoted term cannot cover itself.

Question sets: complete = queries fully recovered from an untruncated B0 context (344);
all = 451 with regenerated queries for the rest.
Usage:
  python -I e1/analysis/retrieval_eval.py calibrate                     # proxy vs B1 correctness
  python -I e1/analysis/retrieval_eval.py table --domains web --variants V,Vfull,T:basic --set complete
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.anchors import anchors  # noqa: E402
from e1.common.io import iter_jsonl, load_rows  # noqa: E402
from e1.common.stats import sign_test  # noqa: E402
from e1.pipeline.retrieve import E1, ScoreBank, base_methods, do_score, do_select, entry_ids, frozen_bundles, meta  # noqa: E402

TYPES = ["static-environment", "dynamic-environment", "procedure", "errors-gotchas",
         "static-environment-abs", "dynamic-environment-abs", "procedure-abs"]
SHORT = {"static-environment": "static", "dynamic-environment": "dynamic", "procedure": "procedure",
         "errors-gotchas": "gotchas", "static-environment-abs": "static-abs",
         "dynamic-environment-abs": "dynamic-abs", "procedure-abs": "procedure-abs"}


def _official_metrics():
    if str(Path(__file__).resolve().parents[2]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evaluation.qa_eval_metrics import normalize_phrase, parse_eval_function_spec, split_phrases
    return normalize_phrase, parse_eval_function_spec, split_phrases


def answer_phrases(m: dict) -> list[str] | None:
    """Phrases for answer coverage, or None if the question is not eligible."""
    if m["is_abstention_problem"]:
        return None
    if not m["eval_function"].startswith("norm_phrase_set_match"):
        return None
    _, parse, split = _official_metrics()
    _, kw = parse(m["eval_function"])
    seps = kw.get("separators", [",", ";"])
    norm_kw = {k: kw[k] for k in ("lower", "normalize_hyphen", "strip_punct") if k in kw}
    ph = split(m["answer_gold"], separators=seps, **norm_kw)
    return ph or None


class Content:
    """Lazy per-entry texts: lower-cased (anchors) and officially normalised (answer phrases)."""

    def __init__(self, dom: str):
        self.dom = dom
        self.ids = entry_ids(dom)
        self.idx = {e: i for i, e in enumerate(self.ids)}
        self.low: dict[str, str] = {}
        self.norm: dict[str, str] = {}
        self._need: set[str] = set()

    def load(self, eids: set[str]) -> None:
        eids = {e for e in eids if e not in self.low}
        if not eids:
            return
        normalize_phrase = _official_metrics()[0]
        with open(E1 / f"index/agentrunbook_r_{self.dom}_small/memory_state/pools/raw_state_pool.jsonl") as fp:
            for line in fp:
                eid = line[line.index('"entry_id": "') + 13:].split('"', 1)[0]
                if eid not in eids:
                    continue
                e = json.loads(line)
                t = "\n".join([e["goal"], e["full_action_sequence"], e["slice_action_sequence"], "\n".join(e["slice_urls"]),
                               e["slice_axtree_text"]])
                self.low[eid] = t.lower()
                self.norm[eid] = " " + normalize_phrase(t) + " "


def label_question(m: dict, blocks: list[str], C: Content) -> dict:
    out = {}
    ph = answer_phrases(m)
    if ph is not None:
        union = " ".join(C.norm[b] for b in blocks)
        found = [p for p in ph if f" {p} " in union]
        out["ans"] = len(found) == len(ph)
        out["ans_share"] = len(found) / len(ph)
    a = anchors(m["question_text"])
    if a:
        lows = [C.low[b] for b in blocks]
        out["anc"] = any(all(x in t for x in a) for t in lows)
        out["anc_any"] = any(any(x in t for x in a) for t in lows)
    return out


def question_set(dom: str, which: str) -> list[str]:
    B = frozen_bundles(dom)
    return sorted(q for q, b in B.items() if which == "all" or b["complete"])


def eval_variant(dom: str, variant: str, source: str, qids: list[str], C: Content, bank=None, ids=None) -> dict:
    # coverage does not depend on the rendered similarity, so the V cosine is not needed here
    rows = {r["question_id"]: r for r in do_select(dom, variant, source, None, bank=bank, ids=ids, sim_variant=None)}
    M = meta(dom)
    need = {b["entry_id"] for q in qids for b in rows[q]["raw_blocks"]}
    C.load(need)
    return {q: {**label_question(M[q], [b["entry_id"] for b in rows[q]["raw_blocks"]], C),
                "blocks": [b["entry_id"] for b in rows[q]["raw_blocks"]]} for q in qids}


def summarize(L: dict, qids: list[str], M: dict) -> dict:
    """Coverage rates overall and per question type. primary = ans rate (the proxy that passed the
    B1 calibration: covered 111/128 vs not covered 33/48 correct, Fisher p = 0.0085; anc did not)."""
    res = {}
    for t in ["ALL"] + TYPES:
        S = [q for q in qids if t == "ALL" or M[q]["question_type"] == t]
        a = [L[q]["ans"] for q in S if "ans" in L[q]]
        c = [L[q]["anc"] for q in S if "anc" in L[q]]
        res[t] = {"ans": (sum(a), len(a)), "anc": (sum(c), len(c))}
    a, c = res["ALL"]["ans"], res["ALL"]["anc"]
    res["primary"] = a[0] / a[1] if a[1] else 0.0
    return res


def paired(Lx: dict, Lv: dict, qids: list[str], key: str) -> tuple[int, int, float]:
    S = [q for q in qids if key in Lx[q] and key in Lv[q]]
    gain = sum(Lx[q][key] and not Lv[q][key] for q in S)
    loss = sum(Lv[q][key] and not Lx[q][key] for q in S)
    return gain, loss, sign_test(gain, loss)


def fmt(x: tuple[int, int]) -> str:
    return f"{x[0]}/{x[1]}" if x[1] else "-"


STRATA = ["qtype", "premise", "anchors", "qidf", "pos", "plen", "rarity"]  # fixed in analysis/strata.py


def two_by_two(Lx: dict, Lv: dict, S: list[str], key: str) -> dict:
    S = [q for q in S if key in Lx[q] and key in Lv[q]]
    both = sum(Lx[q][key] and Lv[q][key] for q in S)
    gain = sum(Lx[q][key] and not Lv[q][key] for q in S)
    loss = sum(Lv[q][key] and not Lx[q][key] for q in S)
    n = len(S)
    net = (gain - loss) / n if n else 0.0
    # 95% CI of the mean paired difference d in {-1, 0, 1} (normal approximation)
    var = ((gain + loss) / n - net ** 2) if n else 0.0
    half = 1.96 * (max(var, 0.0) / n) ** 0.5 if n > 1 else 0.0
    return {"n": n, "both": both, "gain": gain, "loss": loss, "neither": n - both - gain - loss,
            "net": gain - loss, "ci": (net - half, net + half), "p": sign_test(gain, loss)}


def strata_report(dom: str, L: dict, variants: list[str], ref: str, qids: list[str], key: str) -> None:
    sp = E1 / "e2work/strata" / f"{dom}.json"
    if not sp.exists():
        print(f"(no strata file {sp}; run e1/analysis/strata.py)")
        return
    ST = json.loads(sp.read_text())
    print(f"-- {key}: 2x2 vs {ref} by pre-registered stratum: n both/gain/loss/neither  net  [95% CI of net rate]  sign p")
    for v in variants:
        if v == ref:
            continue
        print(f"   [{v}]")
        for k in STRATA:
            labs = sorted({ST[q][k] for q in qids if k in ST.get(q, {})})
            for lab in labs:
                S = [q for q in qids if ST.get(q, {}).get(k) == lab]
                r = two_by_two(L[v], L[ref], S, key)
                if r["n"] == 0:
                    continue
                print(f"     {k:8s}={lab:24s} n={r['n']:3d} {r['both']:3d}/{r['gain']:2d}/{r['loss']:2d}/{r['neither']:3d}  "
                      f"net {r['net']:+3d}  [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]  p={r['p']:.3f}")


def do_calibrate(arm: str = "b1_small") -> None:
    """Proxy vs B1 correctness on the complete questions (the reader saw every B0 block)."""
    sc = load_rows(E1 / "runs", arm, ("question_id", "score_bool"))
    for which in ["ans", "anc"]:
        tab = defaultdict(lambda: [0, 0])
        for dom in ["web", "enterprise"]:
            C, M = Content(dom), meta(dom)
            Q = {r["question_id"]: r for r in iter_jsonl(E1 / f"cache/queries/{dom}.jsonl")}
            qids = [q for q in question_set(dom, "complete")]
            C.load({b["entry_id"] for q in qids for b in Q[q]["baseline_raw_blocks"]})
            for q in qids:
                lab = label_question(M[q], [b["entry_id"] for b in Q[q]["baseline_raw_blocks"]], C)
                if which not in lab:
                    continue
                grp = ("normal" if not M[q]["is_abstention_problem"] else "false-premise", lab[which])
                tab[grp][0] += bool(sc[q]["score_bool"]); tab[grp][1] += 1
        print(f"\ncalibration [{which}] on {arm}, complete questions (both domains): accuracy by proxy label")
        for grp in sorted(tab):
            k, n = tab[grp]
            print(f"  {grp[0]:13s} covered={str(grp[1]):5s}  {k}/{n} = {k / n:.3f}")
        for g in ["normal", "false-premise"]:
            a, b = tab[(g, True)], tab[(g, False)]
            if a[1] and b[1]:
                # Fisher exact (two-sided) via hypergeometric, no scipy
                print(f"  {g}: covered {a[0]}/{a[1]} vs not covered {b[0]}/{b[1]}; Fisher p = {fisher(a[0], a[1] - a[0], b[0], b[1] - b[0]):.4f}")


def fisher(a: int, b: int, c: int, d: int) -> float:
    from math import comb
    n1, n2, k = a + b, c + d, a + c
    tot = comb(n1 + n2, k)
    p0 = comb(n1, a) * comb(n2, c) / tot
    p = 0.0
    for x in range(max(0, k - n2), min(k, n1) + 1):
        px = comb(n1, x) * comb(n2, k - x) / tot
        if px <= p0 * (1 + 1e-9):
            p += px
    return min(1.0, p)


def do_table(domains: list[str], variants: list[str], source: str, which: str, ref: str = "V", out_json: Path | None = None) -> dict:
    allres = {}
    for dom in domains:
        need = sorted({m for v in variants + [ref] for m in base_methods(v)})
        do_score(dom, list(dict.fromkeys(need)))
        C, M, bank = Content(dom), meta(dom), ScoreBank(dom)
        ids = C.ids
        qids = question_set(dom, which)
        # a variant may carry its own query source: "<variant>@<frozen|question|both>"
        L = {v: eval_variant(dom, v.split("@")[0], v.split("@")[1] if "@" in v else source, qids, C, bank, ids) for v in variants}
        if ref not in L:
            L[ref] = eval_variant(dom, ref, source, qids, C, bank, ids)
        print(f"\n=== {dom}, set={which} (n={len(qids)}), query source={source}")
        hdr = f"{'variant':28s} {'primary':>7s} " + " ".join(f"{SHORT.get(t, t)[:11]:>11s}" for t in ["ALL"] + TYPES)
        for key in ["ans", "anc"]:
            print(f"-- {key}: covered/eligible")
            print(hdr)
            for v in variants:
                s = summarize(L[v], qids, M)
                print(f"{v:28s} {s['primary']:7.3f} " + " ".join(f"{fmt(s[t][key]):>11s}" for t in ["ALL"] + TYPES))
            print(f"-- {key}: paired vs {ref}: gain/loss (sign-test p)")
            for v in variants:
                if v == ref:
                    continue
                cells = []
                for t in ["ALL"] + TYPES:
                    S = [q for q in qids if t == "ALL" or M[q]["question_type"] == t]
                    g, l, p = paired(L[v], L[ref], S, key)
                    cells.append(f"{g}/{l}" + (f"({p:.2f})" if g + l else ""))
                print(f"{v:28s} " + " ".join(f"{c:>11s}" for c in cells))
        for key in ["ans", "anc"]:
            strata_report(dom, L, variants, ref, qids, key)
        allres[dom] = {v: summarize(L[v], qids, M) for v in L}
        allres[dom]["_labels"] = {v: {q: {k: L[v][q][k] for k in L[v][q] if k != "blocks"} for q in qids} for v in L}
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(allres))
    return allres


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["calibrate", "table"])
    ap.add_argument("--domains", default="web")
    ap.add_argument("--variants", default="V,Vfull,T:basic,G:both:basic")
    ap.add_argument("--source", default="frozen")
    ap.add_argument("--set", default="complete", choices=["complete", "all"])
    ap.add_argument("--ref", default="V")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.what == "calibrate":
        do_calibrate()
    else:
        do_table(a.domains.split(","), a.variants.split(","), a.source, a.set, a.ref, Path(a.json).expanduser() if a.json else None)


if __name__ == "__main__":
    main()
