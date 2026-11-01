#!/bin/bash
#SBATCH -p gpu-long --gpus=h100-96:1 -c 16 --mem=96G --time=05:00:00
#SBATCH -J v2-keys -o %x.%j.log
#SBATCH --exclude=xgpj0
# TRAD v2 step keys (retrieval only, no reader run). Resumable: rerun continues where it stopped.
# Pre-registration: CP5101/experiments/trad-v2-plan-2026-11-02.md
set -uo pipefail
killsrv() { for p in ${READER_PID:-} ${EMBED_PID:-}; do [ -n "$p" ] && [ "$p" != 0 ] && kill "$p" 2>/dev/null; done; return 0; }
trap 'killsrv; exit 1' TERM INT
E1=$HOME/e1; REPO=${REPO:-$E1/LongMemEval-V2}; export E1
export READER_MODEL=Qwen/Qwen3.5-9B LME_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
export HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENAI_API_KEY=EMPTY
export READER_MAX_LEN=${READER_MAX_LEN:-20480} READER_MAX_SEQS=${READER_MAX_SEQS:-48}
PY=$E1/miniforge3/envs/lme-v2-release/bin/python
cd "$REPO"
source "$REPO/e1/slurm/serve_vllm.sh" || { echo "[v2] vLLM failed"; killsrv; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
rc=0
for d in ${DOMAINS:-web enterprise}; do
  "$PY" -I e1/oneoff/step_keys.py write --domains $d --workers ${WORKERS:-48} || rc=1
done
[ $rc = 0 ] && "$PY" -I e1/oneoff/step_keys.py embed --workers 8 || rc=1
[ $rc = 0 ] && "$PY" -I e1/oneoff/step_keys.py score || rc=1
killsrv
exit $rc
