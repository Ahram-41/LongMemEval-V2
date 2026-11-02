#!/bin/bash
#SBATCH -p gpu-long --gpus=a100-80:1 -c 16 --mem=96G --time=04:00:00
#SBATCH -J e1-si -o %x.%j.log
#SBATCH --exclude=xgpj0
# Self-Index-style key evolution on the AR-R raw-state pool (retrieval only, no reader run). Resumable per iteration.
# Pre-registration: CP5101/experiments/self-index-plan-2026-11-02.md
# Env: DOMAINS (default web), ITERS (default 5), WORKERS (default 32), SISIM task|ui, SITAG (work dir), SIMAT (matrix name), EVAL_VARIANTS
set -uo pipefail
killsrv() { for p in ${READER_PID:-} ${EMBED_PID:-}; do [ -n "$p" ] && [ "$p" != 0 ] && kill "$p" 2>/dev/null; done; return 0; }
trap 'killsrv; exit 1' TERM INT
E1=$HOME/e1; REPO=${REPO:-$E1/LongMemEval-V2}; export E1
export READER_MODEL=Qwen/Qwen3.5-9B LME_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
export HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENAI_API_KEY=EMPTY
export SISIM=${SISIM:-task} SIKEY=${SIKEY:-need} SITAG=${SITAG:-si} SIMAT=${SIMAT:-SI}
export READER_MAX_LEN=${READER_MAX_LEN:-32768} READER_MAX_SEQS=${READER_MAX_SEQS:-48}
PY=$E1/miniforge3/envs/lme-v2-release/bin/python
cd "$REPO"
source "$REPO/e1/slurm/serve_vllm.sh" || { echo "[si] vLLM failed"; killsrv; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
rc=0
"$PY" -I e1/oneoff/self_index.py iterate --domains ${DOMAINS:-web} --iters ${ITERS:-5} --workers ${WORKERS:-32} || rc=1
"$PY" -I e1/oneoff/self_index.py score --domains ${DOMAINS:-web} || rc=1
killsrv
[ $rc = 0 ] && "$PY" -I e1/oneoff/step_keys.py eval --evidence $E1/v2work/evidence.json --variants "V,SI,${SIMAT:-SI},${SIMAT:-SI}k" --out $E1/v2work/reports/eval_${SITAG:-si}_${DOMAINS:-web}.json || rc=1
exit $rc
