"""Oracle upper bound for "answer page not in the reader context".

Two sources of gold pages, merged into one arm (retrieved/orc):
  A. phrase-pinned: waterfall stages 3 / 3or4 whose answer phrases pin <= MAX_GOLD raw-state entries
     (e2work/oracle/gold_entries.json); insert the top-K such entries by official V cosine to any frozen query.
  B. evidence rules (--evidence, the TRAD v2 page-level rules of the fork session, commit 88f0f84): for every
     evidence group pick the entry satisfying the rule with the highest V cosine; one insert per group (<= 3).
     Where a question has both, B wins.
Inserted entries go to the FRONT of the raw-state blocks; the same number of blocks is dropped at the END, so
block count, notes, events, prompt and reader are the B1 recipe. Writes retrieved/orc/<dom>.jsonl (all
questions; only selected ones differ from e2_v), oracle/ids.txt and oracle/plan.json.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.pipeline.retrieve import WORK, ScoreBank, entry_ids  # noqa: E402

MAX_GOLD = 60
K = 2
MAX_GROUPS = 3


def best_by_v(bank, pos, queries, entries):
    out = []
    for e in entries:
        s, qq = max((float(bank.scores("V", qq)[pos[e]]), qq) for qq in queries)
        out.append((s, qq, e))
    out.sort(reverse=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=None, help="TRAD v2 evidence rules json ({note, questions:{qid:{domain, groups}}})")
    a = ap.parse_args()
    gold = json.load(open(WORK / "oracle/gold_entries.json"))
    ev_sets, ev_rules = {}, {}
    if a.evidence:
        from e1.oneoff.step_keys import evidence_sets
        ev_rules = json.load(open(a.evidence))["questions"]
        ev_sets = evidence_sets(ev_rules)
        print(f"[oracle] evidence rules: {len(ev_rules)} questions; groups with >=1 entry in pool: "
              f"{sum(len(s) > 0 for G in ev_sets.values() for s in G.values())}/{sum(len(G) for G in ev_sets.values())}")
    plan, ids = {}, []
    for dom in ("web", "enterprise"):
        bank = ScoreBank(dom)
        store = entry_ids(dom)
        pos = {e: i for i, e in enumerate(store)}
        out = WORK / "retrieved/orc" / f"{dom}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out, "w") as fp:
            for r in iter_jsonl(WORK / f"retrieved/e2_v/{dom}.jsonl"):
                q = r["question_id"]
                have = {b["entry_id"] for b in r["raw_blocks"]}
                pick, src = [], None
                if q in ev_sets:
                    src = "evidence"
                    for g, ents in list(ev_sets[q].items())[:MAX_GROUPS]:
                        if ents & have:
                            continue  # this group is already covered by a B1 block
                        c = best_by_v(bank, pos, r["queries"], sorted(ents))
                        if c:
                            pick.append(c[0] + (g,))
                else:
                    g = gold.get(q)
                    if g and 0 < g["n_gold_entries"] <= MAX_GOLD:
                        src = "phrase"
                        c = best_by_v(bank, pos, r["queries"], [e for e in g["gold_entries"] if e not in have])
                        pick = [x + ("phrase",) for x in c[:K]]
                if pick:
                    keep = r["raw_blocks"][:max(0, len(r["raw_blocks"]) - len(pick))]
                    new = [{"query": qq, "rank": i + 1, "score": s, "method_score": s, "entry_id": e, "oracle": g}
                           for i, (s, qq, e, g) in enumerate(pick)] + keep
                    w = gold.get(q, {})
                    plan[q] = {"domain": dom, "source": src, "stage": w.get("stage"), "b1_correct": w.get("correct"),
                               "n_gold": w.get("n_gold_entries"), "inserted": [(e, g, round(s, 4)) for s, qq, e, g in pick],
                               "dropped": [b["entry_id"] for b in r["raw_blocks"][len(keep):]],
                               "groups_total": len(ev_sets.get(q, {})),
                               "groups_already_in_b1": sum(bool(s & have) for s in ev_sets.get(q, {}).values())}
                    r = {**r, "variant": "oracle+V", "raw_blocks": new}
                    ids.append(q)
                    n += 1
                fp.write(json.dumps(r, ensure_ascii=True) + "\n")
        print(f"[oracle] {dom}: {n} questions modified -> {out}")
    (WORK / "oracle/plan.json").write_text(json.dumps(plan, indent=1))
    (WORK / "oracle/ids.txt").write_text(",".join(sorted(ids)))
    print("selected", len(ids), Counter((p["source"], p["b1_correct"]) for p in plan.values()))
    skipped = [q for q in ev_sets if q not in plan]
    print("evidence questions with nothing to insert (all groups already in B1 or no pool entry):", skipped)


if __name__ == "__main__":
    main()
