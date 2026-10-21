#!/bin/bash
# Re-score a finished run with the LLM judge only: reuses the saved index, cached
# prompt_rows.jsonl and reader_outputs_cache.jsonl, so no vLLM/GPU is needed.
# transformers 5.x cannot import on CPU-only nodes ("Cannot access accelerator"), so run on the
# cheapest GPU class; the GPU is not used for compute:
#   sbatch -p gpu --gpus=nv:1 -c 4 --mem=32G --time=00:45:00 -J e1-score -o %x.%j.log --wrap "bash $HOME/e1/LongMemEval-V2/e1/slurm/score_only.sh"
set -euo pipefail
E1=$HOME/e1; REPO=$E1/LongMemEval-V2
METHOD=${METHOD:-agentrunbook_r}; RUN_TAG=${RUN_TAG:-${METHOD}_small}; DOMAINS=${DOMAINS:-"web enterprise"}
[ -f "$E1/.env" ] || { echo "missing $E1/.env with OPENAI_API_KEY"; exit 2; }
set -a; source "$E1/.env"; set +a
export DATA_ROOT=$E1/data/longmemeval-v2 HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
PY=$E1/miniforge3/envs/lme-v2-release/bin/python
cd "$REPO"
for d in $DOMAINS; do
  out=$E1/runs/$RUN_TAG/${d}_small
  for f in prompt_rows.jsonl reader_outputs_cache.jsonl; do [ -s "$out/$f" ] || { echo "$d: missing $out/$f"; exit 3; }; done
  rm -f "$out/aggregated_metrics.json"
  "$PY" e1/pipeline/run_e1.py --index-root "$E1/index" --method "$METHOD" --domain "$d" --tier small \
     --data-root "$DATA_ROOT" --output-dir "$out" --reader-max-concurrent-requests 8 --prompt-build-max-workers 8
done
"$PY" leaderboard/combine_aggregated_metrics.py \
   "$E1/runs/$RUN_TAG/enterprise_small/aggregated_metrics.json" "$E1/runs/$RUN_TAG/web_small/aggregated_metrics.json" \
   -o "$E1/runs/$RUN_TAG/combined_metrics.json"
cat "$E1/runs/$RUN_TAG/combined_metrics.json"
