"""Oracle upper bound for "answer page not in the reader context" (waterfall stages 3 / 3or4).

For every such question whose answer phrases pin down <= MAX_GOLD raw-state entries, insert the
best such entries (highest official V cosine to any of the frozen queries of that question) at the
FRONT of the raw-state blocks and drop the same number of blocks at the END (block count unchanged).
Everything else (notes, events, prompt, reader) is the B1 recipe. Writes retrieved/orc/<dom>.jsonl
(all questions; only the selected ones differ from e2_v) plus oracle/ids.txt and oracle/plan.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e1.common.io import iter_jsonl  # noqa: E402
from e1.pipeline.retrieve import WORK, ScoreBank, entry_ids  # noqa: E402

MAX_GOLD = 60
K = 2


def main() -> None:
    gold = json.load(open(WORK / "oracle/gold_entries.json"))
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
                g = gold.get(q)
                if g and 0 < g["n_gold_entries"] <= MAX_GOLD:
                    have = {b["entry_id"] for b in r["raw_blocks"]}
                    cands = []
                    for e in g["gold_entries"]:
                        if e in have:
                            continue
                        s, qq = max((float(bank.scores("V", qq)[pos[e]]), qq) for qq in r["queries"])
                        cands.append((s, qq, e))
                    cands.sort(reverse=True)
                    pick = cands[:K]
                    if pick:
                        keep = r["raw_blocks"][:max(0, len(r["raw_blocks"]) - len(pick))]
                        new = [{"query": qq, "rank": i + 1, "score": s, "method_score": s, "entry_id": e, "oracle": True}
                               for i, (s, qq, e) in enumerate(pick)] + keep
                        plan[q] = {"domain": dom, "stage": g["stage"], "b1_correct": g["correct"], "n_gold": g["n_gold_entries"],
                                   "inserted": [(e, round(s, 4)) for s, qq, e in pick],
                                   "dropped": [b["entry_id"] for b in r["raw_blocks"][len(keep):]]}
                        r = {**r, "variant": "oracle+V", "raw_blocks": new}
                        ids.append(q)
                        n += 1
                fp.write(json.dumps(r, ensure_ascii=True) + "\n")
        print(f"[oracle] {dom}: {n} questions modified -> {out}")
    (WORK / "oracle/plan.json").write_text(json.dumps(plan, indent=1))
    (WORK / "oracle/ids.txt").write_text(",".join(sorted(ids)))
    print("selected", len(ids), Counter((p["stage"], p["b1_correct"]) for p in plan.values()))


if __name__ == "__main__":
    main()
