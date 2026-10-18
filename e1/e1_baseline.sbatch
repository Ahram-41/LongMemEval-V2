#!/bin/bash
#SBATCH -p gpu-long --gpus=h100-96:1 -c 16 --mem=96G --time=2-23:00:00
#SBATCH -J e1-baseline -o %x.%j.log
#SBATCH --signal=B:TERM@180
# xgpj0 is aarch64; our conda envs are x86_64
#SBATCH --exclude=xgpj0
# Resumable RAG slice+notes baseline on LME-V2 Small. Re-submit the same script to resume;
# on SIGTERM (time limit) it resubmits itself. Override: LIMIT=5 for a smoke test,
# DOMAINS="web enterprise", RUN_TAG=..., READER_MAX_LEN=65536.
set -uo pipefail
# never "kill 0": that signals the whole process group, including this script
killsrv() { for p in ${READER_PID:-} ${EMBED_PID:-}; do [ -n "$p" ] && [ "$p" != 0 ] && kill "$p" 2>/dev/null; done; return 0; }
E1=$HOME/e1; REPO=$E1/LongMemEval-V2
export E1
METHOD=${METHOD:-agentrunbook_r}
RUN_TAG=${RUN_TAG:-${METHOD}_small}
DOMAINS=${DOMAINS:-"web enterprise"}
LIMIT=${LIMIT:-}
SMOKE_TRAJ=${SMOKE_TRAJ:-}
[ -f "$E1/.env" ] && set -a && source "$E1/.env" && set +a   # OPENAI_API_KEY for the judge
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export READER_MODEL=Qwen/Qwen3.5-9B
export LME_CONTROLLER_MODEL=Qwen/Qwen3.5-9B
export LME_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
export DATA_ROOT=$E1/data/longmemeval-v2 HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
LME_PY=$E1/miniforge3/envs/lme-v2-release/bin/python

RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}; RESUBMIT_MAX=${RESUBMIT_MAX:-3}
resubmit() {
  killsrv
  if [ "$RESUBMIT_COUNT" -ge "$RESUBMIT_MAX" ]; then echo "[sbatch] resubmit cap $RESUBMIT_MAX reached; NOT resubmitting"; exit 1; fi
  echo "[sbatch] time limit, resubmitting ($((RESUBMIT_COUNT+1))/$RESUBMIT_MAX)"
  cd "$E1" && RESUBMIT_COUNT=$((RESUBMIT_COUNT+1)) sbatch --export=ALL "$REPO/e1/e1_baseline.sbatch"; exit 0
}
trap resubmit TERM

source "$REPO/e1/serve_vllm.sh" || { echo "[sbatch] vLLM failed to start"; exit 1; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

cd "$REPO"
status=0
for d in $DOMAINS; do
  out=$E1/runs/${RUN_TAG}${LIMIT:+_limit$LIMIT}${SMOKE_TRAJ:+_traj$SMOKE_TRAJ}/${d}_small
  "$LME_PY" e1/run_e1.py --index-root "$E1/index" \
     --method "$METHOD" --domain "$d" --tier small \
     --data-root "$DATA_ROOT" --output-dir "$out" \
     --reader-max-concurrent-requests ${READER_MAX_SEQS:-8} --prompt-build-max-workers 8 \
     ${LIMIT:+--limit $LIMIT} ${SMOKE_TRAJ:+--smoke-trajectories $SMOKE_TRAJ} &
  wait $!   # 'wait' (not foreground exec) so the TERM trap can fire
  rc=$?; [ $rc -ne 0 ] && { echo "[sbatch] domain $d failed rc=$rc"; status=$rc; }
done
if [ "$status" -eq 0 ] && [ -z "$LIMIT" ]; then
  "$LME_PY" leaderboard/combine_aggregated_metrics.py \
     "$E1/runs/$RUN_TAG/enterprise_small/aggregated_metrics.json" \
     "$E1/runs/$RUN_TAG/web_small/aggregated_metrics.json" \
     -o "$E1/runs/$RUN_TAG/combined_metrics.json" && cat "$E1/runs/$RUN_TAG/combined_metrics.json"
fi
killsrv
exit $status
