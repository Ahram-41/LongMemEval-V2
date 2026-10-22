#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G --time=02:50:00
#SBATCH -J e3-embed -o /home/w/wang41/e1/e3work/logs/%x.%j.log
# xgpj0 is aarch64; our conda envs are x86_64
#SBATCH --exclude=xgpj0
# E3 embeddings (LongMemEval_S rounds + questions). Embedding server only; no reader. No self-resubmit;
# shards are resumable, so a manual rerun continues. Submit with an explicit --gpus=<type>:1.
#   STAGE=test  : embed an evenly strided sample of LIMIT pieces (timing) + the 500 queries
#   STAGE=embed : all pieces + queries
set -uo pipefail
killsrv() { for p in ${READER_PID:-} ${EMBED_PID:-}; do [ -n "$p" ] && [ "$p" != 0 ] && kill "$p" 2>/dev/null; done; return 0; }
trap 'killsrv; exit 1' TERM INT
E1=$HOME/e1; REPO=${REPO:-$E1/LongMemEval-V2}; export E1
STAGE=${STAGE:?set STAGE=test|embed}
export HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENAI_API_KEY=EMPTY
export NO_READER=1 EMBED_MEM_FRAC=${EMBED_MEM_FRAC:-0.85} LOGDIR=$E1/e3work/logs
PY=$E1/miniforge3/envs/lme-v2-release/bin/python
cd "$REPO"
source "$REPO/e1/slurm/serve_vllm.sh" || { echo "[e3] vLLM failed"; killsrv; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
rc=0
case "$STAGE" in
  test)  "$PY" -I e1/lmes/embed.py pieces --limit ${LIMIT:-2000} --batch ${BATCH:-32} --workers ${WORKERS:-8} || rc=1 ;;
  embed) "$PY" -I e1/lmes/embed.py queries --batch 32 --workers 4 || rc=1
         "$PY" -I e1/lmes/embed.py pieces --batch ${BATCH:-32} --workers ${WORKERS:-8} || rc=1 ;;
  *) echo "unknown STAGE"; rc=2 ;;
esac
killsrv
exit $rc
