#!/bin/bash
# One-shot analysis once the arms have finished. Safe to rerun; reads only finished per_question files.
R=$HOME/e1/runs; cd $HOME/e1/LongMemEval-V2
for a in pa ps pnf; do
  [ -f $R/${a}_small/web_small/per_question.jsonl ] && [ -f $R/b1_small/web_small/per_question.jsonl ] && \
    python3 e1/merge_noted.py $R b1_small ${a}_small ${a}_merged_small
done
echo "===== paired comparison (reference B0)"
python3 e1/compare_arms.py $R agentrunbook_r_small b0r_small b1_small pa_merged_small ps_merged_small pnf_merged_small
echo "===== paired comparison (reference B1: effect of the note alone)"
python3 e1/compare_arms.py $R b1_small pa_merged_small ps_merged_small pnf_merged_small
echo "===== budget probe"
python3 e1/budget_probe.py $R
for a in pa ps pnf; do for d in web_small enterprise_small; do [ -f $R/${a}_small/$d/per_question.jsonl ] && python3 e1/noted_eff.py ${a}_small $d; done; done
