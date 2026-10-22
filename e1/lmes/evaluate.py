#!/usr/bin/env python3
"""E3 evaluation (CPU, numpy only). Pre-registered rules: CP5101/experiments/E3-log-2026-10-22.md §预注册.

Variants (round-level scores per question; session score = max over its rounds):
  T, Tb      : BM25 camel / basic tokenizer (scores/T_camel.npz, scores/T_basic.npz)
  V          : dense (scores/V.npz)
  rrf10/rrf60: sum over {V, T} of 1 / (k + rank); ranks 1-based over the rounds of the question's
               haystack, ties get the AVERAGE rank
  w<x>       : (1 - x) * minmax(V) + x * minmax(T); min-max over the rounds of the question's haystack
               (constant list -> 0). x = BM25 weight; w0 ranks like V, w1 like T.
Fusion is done at round level, then max per session.

Ranking ties: sessions by (-score, sha256(session_id)); rounds by (-score, sha256(qid:position)).
Metrics per question: session recall_all@k / recall_any@k / recall_frac@k (k = 5, 10) against
answer_session_ids; turn_all@10 / turn_any@10 over rounds that contain a has_answer turn (questions without
such a turn are excluded from the turn metric). Abstention (_abs) questions are reported separately only.

  table   --set dev --round NAME --variants T,V,... [--pairs V:T,rrf10:T,...]
  verdict --set dev|heldout --round NAME --fusion VARIANT
held-out runs only when e1/lmes/E3_FROZEN.json is committed and e1/ is clean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.common.manifest import git_version, write_manifest  # noqa: E402
from e1.common.stats import sign_test  # noqa: E402
from e1.lmes.prep import PREP, WORK  # noqa: E402
from e1.lmes.split import load_split  # noqa: E402
from e1.retrieval.fuse import minmax  # noqa: E402

TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning"]
ABBR = {"single-session-user": "SSU", "single-session-assistant": "SSA", "single-session-preference": "SSP",
        "multi-session": "MS", "knowledge-update": "KU", "temporal-reasoning": "TR"}
KS = (5, 10)
WGRID = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1]
FUSIONS = ["rrf10", "rrf60"] + [f"w{w:g}" for w in WGRID if 0 < w < 1]
REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "e1/lmes/E3_FROZEN.json"
METRICS = [f"all@{k}" for k in KS] + [f"any@{k}" for k in KS] + [f"frac@{k}" for k in KS] + ["turn_all@10", "turn_any@10"]
# Supplementary set, added after the BM25 dev result and before any dense result (E3 log 12:36); reported
# separately, never used by the R1-R4 verdicts.
SUPP_KS, SUPP_TURN_KS = (1, 3), (5, 10, 20)
SUPP = [f"{a}@{k}" for k in SUPP_KS for a in ("all", "any")] + [f"turn_{a}@{k}" for k in SUPP_TURN_KS for a in ("all", "any")]


def h64(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


# ---------------------------------------------------------------- stats
def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """2x2 [[a, b], [c, d]]; sum of hypergeometric probabilities <= observed."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def p(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))


def wilcoxon(d) -> float:
    """Two-sided signed-rank test, normal approximation with tie correction and continuity; zeros dropped."""
    d = np.asarray([x for x in d if abs(x) > 1e-12])
    n = len(d)
    if n == 0:
        return 1.0
    a = np.abs(d)
    order = np.argsort(a)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    wp = ranks[d > 0].sum()
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - (cnt ** 3 - cnt).sum() / 48
    if var <= 0:
        return 1.0
    z = (abs(wp - mu) - 0.5) / math.sqrt(var)
    return float(math.erfc(max(z, 0) / math.sqrt(2)))


def avg_rank_desc(s: np.ndarray) -> np.ndarray:
    """1-based ranks, best = 1, ties -> average rank."""
    order = np.argsort(-s, kind="stable")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(-s, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / cnt)[inv]


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, ids: set[str] | None = None):
        self.q = {}
        for r in iter_jsonl(PREP / "questions.jsonl"):
            if ids is not None and r["question_id"] not in ids:
                continue
            for k in ("question", "answer", "haystack_dates", "r_uid", "r_turns", "ev_turns"):
                r.pop(k, None)
            r["r_sess"] = np.array(r["r_sess"]); r["r_ev"] = np.array(r["r_ev"], dtype=bool)
            r["sess_tb"] = np.array([h64(s) for s in r["haystack_session_ids"]], dtype=np.uint64)
            r["round_tb"] = np.array([h64(f"{r['question_id']}:{i}") for i in range(len(r["r_sess"]))], dtype=np.uint64)
            self.q[r["question_id"]] = r
        self.base: dict[str, dict] = {}

    def base_scores(self, name: str) -> dict:
        if name not in self.base:
            f = {"T": "T_camel", "Tb": "T_basic", "V": "V"}[name]
            z = np.load(WORK / "scores" / f"{f}.npz")
            self.base[name] = {k: z[k].astype(np.float64) for k in self.q}
        return self.base[name]

    def variant(self, name: str, qid: str) -> np.ndarray:
        if name in ("T", "Tb", "V"):
            return self.base_scores(name)[qid]
        v, t = self.base_scores("V")[qid], self.base_scores("T")[qid]
        if name.startswith("rrf"):
            k = float(name[3:])
            return 1 / (k + avg_rank_desc(v)) + 1 / (k + avg_rank_desc(t))
        if name.startswith("w"):
            w = float(name[1:])
            return (1 - w) * minmax(v) + w * minmax(t)
        raise ValueError(name)


def evaluate_one(q: dict, rs: np.ndarray) -> dict:
    n_s = len(q["haystack_session_ids"])
    ss = np.full(n_s, -np.inf)
    np.maximum.at(ss, q["r_sess"], rs)
    order = np.lexsort((q["sess_tb"], -ss))
    pos = np.empty(n_s, dtype=int); pos[order] = np.arange(1, n_s + 1)
    sid_pos = {s: int(pos[i]) for i, s in enumerate(q["haystack_session_ids"])}
    ans = list(dict.fromkeys(q["answer_session_ids"]))
    ev_rank = [sid_pos.get(s) for s in ans]
    out = {"ev_session_ranks": ev_rank, "n_ev_sessions": len(ans),
           "top10": [[q["haystack_session_ids"][i], round(float(ss[i]), 5)] for i in order[:10]]}
    for k in sorted(set(KS) | set(SUPP_KS)):
        hit = [r is not None and r <= k for r in ev_rank]
        out[f"all@{k}"] = int(bool(hit) and all(hit)); out[f"any@{k}"] = int(any(hit))
        out[f"frac@{k}"] = (sum(hit) / len(hit)) if hit else None
        out[f"tie@{k}"] = int(n_s > k and ss[order[k - 1]] == ss[order[k]])
        if not hit:
            out[f"all@{k}"] = out[f"any@{k}"] = None
    ro = np.lexsort((q["round_tb"], -rs))
    rpos = np.empty(len(rs), dtype=int); rpos[ro] = np.arange(1, len(rs) + 1)
    evr = rpos[q["r_ev"]]
    out["ev_round_ranks"] = sorted(int(x) for x in evr)
    for k in SUPP_TURN_KS:
        out[f"turn_all@{k}"] = int((evr <= k).all()) if len(evr) else None
        out[f"turn_any@{k}"] = int((evr <= k).any()) if len(evr) else None
    return out


def run(D: Data, variants: list[str]) -> dict:
    return {v: {qid: evaluate_one(q, D.variant(v, qid)) for qid, q in D.q.items()} for v in variants}


# ---------------------------------------------------------------- reporting
def groups(D: Data) -> dict[str, list[str]]:
    g = defaultdict(list)
    for qid, q in D.q.items():
        g[(ABBR[q["question_type"]] + "_abs") if q["is_abs"] else ABBR[q["question_type"]]].append(qid)
    g["ALL(non-abs)"] = [qid for qid, q in D.q.items() if not q["is_abs"]]
    return g


def mean_of(res, ids, m):
    xs = [res[q][m] for q in ids if res[q][m] is not None]
    return (sum(xs) / len(xs), len(xs), sum(xs)) if xs else (float("nan"), 0, 0)


def paired(resA, resB, ids, m):
    """A vs B on metric m: gains (A>B), losses (A<B), sign p, mean diff, SE, Wilcoxon p (frac metrics)."""
    d = [resA[q][m] - resB[q][m] for q in ids if resA[q][m] is not None and resB[q][m] is not None]
    g, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
    n = len(d)
    md = sum(d) / n if n else float("nan")
    se = (np.std(d, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "gain": g, "loss": l, "p_sign": sign_test(g, l), "mean_diff": md, "se": float(se),
            "p_wilcoxon": wilcoxon(d)}


ORDER = ["SSU", "SSA", "SSP", "MS", "KU", "TR", "ALL(non-abs)"]


def table_text(D, R, variants, pairs) -> tuple[str, list[str]]:
    g = groups(D)
    keys = ORDER + sorted(k for k in g if k.endswith("_abs"))
    lines, grid = [], ["group\tmetric\tvariant\tmean\thits\tn"]
    for m in METRICS + [x for x in SUPP if x not in METRICS]:
        tag = "" if m in METRICS else "  [补充指标, not used by verdicts]"
        lines.append(f"\n== {m} (hits / n per group){tag} ==")
        lines.append("variant".ljust(10) + "".join(k[:14].rjust(16) for k in keys))
        for v in variants:
            row = v.ljust(10)
            for k in keys:
                mu, n, h = mean_of(R[v], g.get(k, []), m)
                row += (f"{h:.4g}/{n} {mu:.3f}" if n else "-").rjust(16)
                grid.append(f"{k}\t{m}\t{v}\t{mu:.4f}\t{h:g}\t{n}")
            lines.append(row)
    for a, b in pairs:
        for m in ("all@5", "all@10", "frac@5", "frac@10", "turn_all@10", "all@1", "all@3", "turn_all@5", "turn_any@5", "turn_all@20"):
            lines.append(f"\n-- {a} vs {b} on {m}: gain/loss (sign p) | mean diff +- SE | wilcoxon p --")
            for k in ORDER:
                r = paired(R[a], R[b], g.get(k, []), m)
                lines.append(f"  {k:<12} n={r['n']:<4} {r['gain']:>3}/{r['loss']:<3} (p={r['p_sign']:.3f})  "
                             f"{r['mean_diff']:+.3f} +- {r['se']:.3f}  wil p={r['p_wilcoxon']:.3f}")
                grid.append(f"{k}\tpair:{m}\t{a}-{b}\t{r['mean_diff']:.4f}\t{r['gain']}/{r['loss']}\t{r['n']}\tp_sign={r['p_sign']:.4f}\tp_wil={r['p_wilcoxon']:.4f}")
    ties = {v: {f"tie@{k}": sum(R[v][q][f"tie@{k}"] for q in g["ALL(non-abs)"]) for k in KS} for v in variants}
    lines.append(f"\nquestions (non-abs) with a score tie across the top-k cutoff: {json.dumps(ties)}")
    return "\n".join(lines) + "\n", grid


def ids_for(setname: str) -> set[str]:
    sp = load_split()
    if setname == "heldout":
        gate_heldout()
    return {q for q, s in sp.items() if s == setname}


def gate_heldout() -> None:
    assert FROZEN.exists(), "held-out needs e1/lmes/E3_FROZEN.json"
    st = subprocess.run(["git", "status", "--porcelain", "--", "e1"], cwd=REPO, capture_output=True, text=True).stdout
    assert not st.strip(), f"e1/ must be clean before the held-out run:\n{st}"
    lg = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(FROZEN)], cwd=REPO, capture_output=True, text=True).stdout
    assert lg.strip(), "E3_FROZEN.json must be committed"


def write_round(name, setname, variants, R, D, text, grid, extra=None):
    d = WORK / "rounds" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "table.txt").write_text(text)
    (d / "grid.tsv").write_text("\n".join(grid) + "\n")
    for v in variants:
        with open(d / f"dump_{v}.jsonl", "w") as fp:
            for qid, q in D.q.items():
                fp.write(json.dumps({"question_id": qid, "type": ABBR[q["question_type"]], "is_abs": q["is_abs"],
                                     "variant": v, **R[v][qid]}) + "\n")
    cfg = {"round": name, "set": setname, "variants": variants, "git": git_version(), **(extra or {})}
    (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    write_manifest(d, inputs={"questions": PREP / "questions.jsonl", "scores": WORK / "scores"}, params=cfg)


# ---------------------------------------------------------------- verdicts
def verdicts(D: Data, R: dict, fusion: str) -> dict:
    g = groups(D)
    out = {"fusion": fusion, "k": {}}
    for k in KS:
        m = f"all@{k}"
        vk = {}
        # R1
        net = {t: paired(R["V"], R["T"], g[t], m) for t in ORDER[:6]}
        sigV = {t for t, r in net.items() if r["p_sign"] < 0.05 and r["gain"] > r["loss"]}
        sigT = {t for t, r in net.items() if r["p_sign"] < 0.05 and r["loss"] > r["gain"]}
        signs = {np.sign(r["gain"] - r["loss"]) for r in net.values()} - {0}
        if (sigV & {"SSP", "MS"}) and not (sigV & {"SSU", "SSA"}):
            r1 = "supported" + (" (with reversal: T significantly better on " + ",".join(sorted(sigT & {"SSU", "SSA"})) + ")" if sigT & {"SSU", "SSA"} else "")
        elif sigV & {"SSU", "SSA"}:
            r1 = "refuted in part (V significantly better on SSU/SSA)"
        elif not sigV and not sigT and len(signs) <= 1:
            r1 = "falsified (no significant per-type difference, no sign reversal)"
        else:
            r1 = "not supported (no significant V gain on SSP/MS)"
        vk["R1"] = {"verdict": r1, "V_minus_T": {t: {x: net[t][x] for x in ("n", "gain", "loss", "p_sign", "mean_diff")} for t in net}}
        # R2
        r2 = {}
        worst = "holds"
        for t in ORDER[:6]:
            best = max(("V", "T"), key=lambda v: mean_of(R[v], g[t], m)[0])
            r = paired(R[fusion], R[best], g[t], m)
            if r["p_sign"] < 0.05 and r["loss"] > r["gain"]:
                s = "falsified"
            elif r["mean_diff"] >= -r["se"] or r["n"] == 0 or math.isnan(r["se"]):
                s = "holds"
            else:
                s = "borderline"
            if s == "falsified" or (s == "borderline" and worst == "holds"):
                worst = s
            r2[t] = {"best_single": best, "status": s, **{x: r[x] for x in ("n", "gain", "loss", "p_sign", "mean_diff", "se")}}
        vk["R2"] = {"verdict": worst, "per_type": r2}
        # R3
        ss_ids = g["SSU"] + g["SSA"] + g["SSP"]
        hard = g["KU"] + g["TR"]
        r3 = {}
        lower = []
        for v in ("T", "V", fusion):
            a, na = mean_of(R[v], hard, m)[2], mean_of(R[v], hard, m)[1]
            c, nc = mean_of(R[v], ss_ids, m)[2], mean_of(R[v], ss_ids, m)[1]
            p = fisher_two_sided(int(a), int(na - a), int(c), int(nc - c))
            is_lower = (a / na < c / nc) and p < 0.05
            lower.append(is_lower)
            per = {t: "%g/%d" % (mean_of(R[v], g[t], m)[2], mean_of(R[v], g[t], m)[1]) for t in ORDER[:6]}
            anyr = {t: "%g/%d" % (mean_of(R[v], g[t], f"any@{k}")[2], mean_of(R[v], g[t], f"any@{k}")[1]) for t in ("MS", "KU", "TR")}
            r3[v] = {"KU+TR": f"{a:g}/{na}", "SS*": f"{c:g}/{nc}", "fisher_p": p, "lower": is_lower, "per_type": per, "any_multi": anyr}
        vk["R3"] = {"verdict": "supported" if all(lower) else ("falsified" if not any(lower) else "partial"), "methods": r3}
        # R4 (on this set; dev vs held-out comparison is done in the results doc)
        r4 = {}
        for t in ORDER[:6]:
            mus = {w: mean_of(R[f"w{w:g}"], g[t], m)[0] for w in WGRID}
            best = max(mus.values())
            W = [w for w in WGRID if abs(mus[w] - best) < 1e-12]
            worst = min(WGRID, key=lambda w: mus[w])
            se = paired(R[f"w{W[0]:g}"], R[f"w{worst:g}"], g[t], m)["se"]   # SE of the paired best-minus-worst difference
            flat = (best - min(mus.values())) <= (se if not math.isnan(se) else 0)
            r4[t] = {"best_w": W, "lean": float(np.median(W)), "curve": {f"{w:g}": round(mus[w], 4) for w in WGRID},
                     "range": round(best - min(mus.values()), 4), "flat_within_1SE": bool(flat)}
        dense_side = max(r4[t]["lean"] for t in ("SSP", "MS"))
        lex_side = min(r4[t]["lean"] for t in ("SSU", "SSA"))
        vk["R4"] = {"ordering_holds_on_this_set": bool(dense_side < lex_side), "max_lean_SSP_MS": dense_side,
                    "min_lean_SSU_SSA": lex_side, "per_type": r4}
        out["k"][str(k)] = vk
    # combine k
    out["combined"] = {}
    for rr in ("R1", "R2", "R3"):
        vs = [out["k"][str(k)][rr]["verdict"] for k in KS]
        out["combined"][rr] = vs[0] if vs[0] == vs[1] else f"k-dependent ({vs[0]} @5; {vs[1]} @10)"
    o = [out["k"][str(k)]["R4"]["ordering_holds_on_this_set"] for k in KS]
    out["combined"]["R4_ordering_on_this_set"] = o[0] if o[0] == o[1] else f"k-dependent ({o[0]} @5; {o[1]} @10)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["table", "verdict"])
    ap.add_argument("--set", default="dev", choices=["dev", "heldout", "all"])
    ap.add_argument("--round", required=True)
    ap.add_argument("--variants", default="T,V")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--fusion", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.set == "all":
        gate_heldout()
        ids = None
    else:
        ids = ids_for(a.set)
    D = Data(ids)
    if a.what == "table":
        variants = a.variants.split(",")
        pairs = [tuple(p.split(":")) for p in a.pairs.split(",") if p]
        R = run(D, variants)
        text, grid = table_text(D, R, variants, pairs)
        text = f"# round {a.round}  set={a.set}  n={len(D.q)}  note: {a.note}\n" + text
        write_round(a.round, a.set, variants, R, D, text, grid, {"note": a.note, "pairs": a.pairs})
        print(text)
    else:
        variants = ["T", "V"] + FUSIONS + ["w0", "w1"]
        variants = list(dict.fromkeys(variants))
        R = run(D, variants)
        fusion = a.fusion
        vd = verdicts(D, R, fusion)
        text, grid = table_text(D, R, variants, [("V", "T"), (fusion, "T"), (fusion, "V")])
        text = f"# round {a.round}  set={a.set}  n={len(D.q)}  verdict run, fusion={fusion}\n" + text
        write_round(a.round, a.set, variants, R, D, text, grid, {"fusion": fusion, "note": a.note})
        (WORK / "rounds" / a.round / "verdicts.json").write_text(json.dumps(vd, indent=2, default=float) + "\n")
        print(text)
        print(json.dumps(vd["combined"], indent=2))


if __name__ == "__main__":
    main()
