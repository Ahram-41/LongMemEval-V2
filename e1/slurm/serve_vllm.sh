#!/bin/bash
# Start Qwen3-Embedding-8B (port 8114) and Qwen3.5-9B reader (port 8023) on the current GPU.
# Usage: source e1/slurm/serve_vllm.sh   (exports EMBED_PID, READER_PID; waits until both are healthy)
# NO_EMBED=1: reader only.  NO_READER=1: embedding only (EMBED_MEM_FRAC can then be raised).
set -u
E1=${E1:-$HOME/e1}
LOGDIR=${LOGDIR:-$E1/logs}; mkdir -p "$LOGDIR"
export HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
READER_MAX_LEN=${READER_MAX_LEN:-262144}   # official protocol: 200k context + 20k output
VLLM_PY=$E1/miniforge3/envs/vllm/bin
export PATH="$VLLM_PY:/usr/local/cuda/bin:$PATH" CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}   # ninja + nvcc for flashinfer JIT
TAG=${SLURM_JOB_ID:-local}
# Per-job ports so two of our jobs on one node never share a server (and never hit a stranger's 8023).
# Two ports per job (reader even, embedding odd) so consecutive job ids never collide
# (the old 20000 + id % 20000 scheme gave job N's embedding port to job N+1's reader).
if [ -n "${SLURM_JOB_ID:-}" ]; then READER_PORT=${READER_PORT:-$((20000 + 2 * (SLURM_JOB_ID % 10000)))}; EMBED_PORT=${EMBED_PORT:-$((READER_PORT + 1))}; fi
READER_PORT=${READER_PORT:-8023}; EMBED_PORT=${EMBED_PORT:-8114}
export READER_PORT EMBED_PORT
export READER_BASE_URL=http://localhost:$READER_PORT/v1 LME_CONTROLLER_BASE_URL=http://localhost:$READER_PORT/v1 LME_EMBEDDING_BASE_URL=http://localhost:$EMBED_PORT/v1

wait_healthy() {  # port, pid, name
  local t=0
  until curl -sf "http://localhost:$1/health" >/dev/null; do
    if ! kill -0 "$2" 2>/dev/null; then echo "[serve] $3 died; see $LOGDIR"; return 1; fi
    sleep 10; t=$((t+10)); [ $t -ge 1800 ] && { echo "[serve] $3 not healthy after 30 min"; return 1; }
  done
  echo "[serve] $3 healthy on :$1 after ${t}s"
}

# Start sequentially: each vLLM instance profiles free memory at startup, so two
# concurrent starts on one GPU under-count each other and fail KV allocation.
if [ -z "${NO_EMBED:-}" ]; then
"$VLLM_PY/vllm" serve Qwen/Qwen3-Embedding-8B --runner pooling --port $EMBED_PORT \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 32 \
  --gpu-memory-utilization ${EMBED_MEM_FRAC:-0.30} \
  > "$LOGDIR/embed.$TAG.log" 2>&1 &
EMBED_PID=$!
export EMBED_PID
wait_healthy $EMBED_PORT "$EMBED_PID" embedding || return 1
else
  EMBED_PID=0; export EMBED_PID; echo "[serve] NO_EMBED=1: reader only"
  READER_MEM_FRAC=${READER_MEM_FRAC:-0.90}
fi

if [ -n "${NO_READER:-}" ]; then READER_PID=0; export READER_PID; echo "[serve] NO_READER=1: embedding only"; return 0; fi
"$VLLM_PY/vllm" serve Qwen/Qwen3.5-9B --port $READER_PORT \
  --dtype bfloat16 --max-model-len "$READER_MAX_LEN" \
  --gpu-memory-utilization ${READER_MEM_FRAC:-0.62} \
  --max-num-seqs ${READER_MAX_SEQS:-8} --enable-prefix-caching \
  --reasoning-parser ${REASONING_PARSER:-qwen3} \
  ${READER_CHAT_TEMPLATE:+--chat-template $READER_CHAT_TEMPLATE} \
  --limit-mm-per-prompt '{"image": '${MAX_IMAGES:-128}'}' \
  > "$LOGDIR/reader.$TAG.log" 2>&1 &
READER_PID=$!
export READER_PID
wait_healthy $READER_PORT "$READER_PID" reader
