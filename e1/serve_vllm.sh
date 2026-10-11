#!/bin/bash
# Start Qwen3-Embedding-8B (port 8114) and Qwen3.5-9B reader (port 8023) on the current GPU.
# Usage: source e1/serve_vllm.sh   (exports EMBED_PID, READER_PID; waits until both are healthy)
set -u
E1=${E1:-$HOME/e1}
LOGDIR=${LOGDIR:-$E1/logs}; mkdir -p "$LOGDIR"
export HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
READER_MAX_LEN=${READER_MAX_LEN:-65536}
VLLM_PY=$E1/miniforge3/envs/vllm/bin
TAG=${SLURM_JOB_ID:-local}

"$VLLM_PY/vllm" serve Qwen/Qwen3-Embedding-8B --runner pooling --port 8114 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.22 \
  --disable-log-requests > "$LOGDIR/embed.$TAG.log" 2>&1 &
EMBED_PID=$!

"$VLLM_PY/vllm" serve Qwen/Qwen3.5-9B --port 8023 \
  --dtype bfloat16 --max-model-len "$READER_MAX_LEN" --gpu-memory-utilization 0.70 \
  --max-num-seqs ${READER_MAX_SEQS:-16} --enable-prefix-caching \
  --reasoning-parser ${REASONING_PARSER:-qwen3} \
  --disable-log-requests > "$LOGDIR/reader.$TAG.log" 2>&1 &
READER_PID=$!
export EMBED_PID READER_PID

wait_healthy() {  # port, pid, name
  local t=0
  until curl -sf "http://localhost:$1/health" >/dev/null; do
    if ! kill -0 "$2" 2>/dev/null; then echo "[serve] $3 died; see $LOGDIR"; return 1; fi
    sleep 10; t=$((t+10)); [ $t -ge 1800 ] && { echo "[serve] $3 not healthy after 30 min"; return 1; }
  done
  echo "[serve] $3 healthy on :$1 after ${t}s"
}
wait_healthy 8114 "$EMBED_PID" embedding && wait_healthy 8023 "$READER_PID" reader
