#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G --time=02:50:00
#SBATCH -J e2-embed -o %x.%j.log
# xgpj0 is aarch64; our conda envs are x86_64
#SBATCH --exclude=xgpj0
# E2 GPU inputs. No self-resubmit; every step is resumable, so a manual rerun continues.
# Submit with an explicit --gpus=<type>:1.
#   STAGE=test  : embedding server only; embed pieces_test.jsonl (5 web trajectories) for timing
#   STAGE=embed : embedding server only; query vectors for the parsed queries, V-full pieces
#                 (both domains), then wait (<= WAIT_REGEN_MIN) for regen DONE markers and embed the
#                 regenerated queries
#   STAGE=regen : controller only (Qwen3.5-9B, official query-bundle call) for incomplete questions
set -uo pipefail
killsrv() { for p in ${READER_PID:-} ${EMBED_PID:-}; do [ -n "$p" ] && [ "$p" != 0 ] && kill "$p" 2>/dev/null; done; return 0; }
trap 'killsrv; exit 1' TERM INT
E1=$HOME/e1; REPO=${REPO:-$E1/LongMemEval-V2}; export E1
STAGE=${STAGE:?set STAGE=test|embed|regen}
export READER_MODEL=Qwen/Qwen3.5-9B LME_CONTROLLER_MODEL=Qwen/Qwen3.5-9B LME_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
export HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENAI_API_KEY=EMPTY
PY=$E1/miniforge3/envs/lme-v2-release/bin/python
cd "$REPO"
case "$STAGE" in
  test|embed) export NO_READER=1 EMBED_MEM_FRAC=${EMBED_MEM_FRAC:-0.85} ;;
  regen) export NO_EMBED=1 READER_MAX_LEN=${READER_MAX_LEN:-65536} ;;
esac
source "$REPO/e1/slurm/serve_vllm.sh" || { echo "[e2] vLLM failed"; killsrv; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
rc=0
case "$STAGE" in
  test)  "$PY" -I e1/pipeline/e2_gpu.py pieces --domains web --pieces-name pieces_test.jsonl --batch ${BATCH:-16} --workers ${WORKERS:-8} || rc=1 ;;
  embed) "$PY" -I e1/pipeline/e2_gpu.py queries || rc=1
         "$PY" -I e1/pipeline/e2_gpu.py pieces --domains ${DOMAINS:-web,enterprise} --batch ${BATCH:-16} --workers ${WORKERS:-8} || rc=1
         t=0; until [ -f "$E1/cache/queries_regen/web.DONE" ] && [ -f "$E1/cache/queries_regen/enterprise.DONE" ]; do
           sleep 60; t=$((t+1)); [ $t -ge ${WAIT_REGEN_MIN:-40} ] && { echo "[e2] regen not done after $t min"; break; }; done
         "$PY" -I e1/pipeline/e2_gpu.py queries || rc=1 ;;
  regen) for d in web enterprise; do
           "$PY" -I e1/pipeline/e2_gpu.py regen --domains $d --workers 8 && touch "$E1/cache/queries_regen/$d.DONE" || rc=1
         done ;;
  *) echo "unknown STAGE"; rc=2 ;;
esac
killsrv
exit $rc
