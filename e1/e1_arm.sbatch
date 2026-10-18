#!/bin/bash
#SBATCH -p gpu-long --gpus=h100-96:1 -c 16 --mem=96G --time=12:00:00
#SBATCH -J e1-arm -o %x.%j.log
# Arms derived from the AgentRunbook-R baseline (retrieval reused byte-for-byte).
#   ARM=gate : premise checker only, on all questions -> gate_summary (no reader)
#   ARM=b1   : official system prompt with the backspace fixed; reader rerun
#   ARM=p    : b1 + premise-check note; reader rerun (reuses gate verdicts if present)
set -uo pipefail
E1=$HOME/e1; REPO=$E1/LongMemEval-V2; export E1
ARM=${ARM:?set ARM=gate|b1|p}; BASE=$E1/runs/agentrunbook_r_small
[ -f "$E1/.env" ] && set -a && source "$E1/.env" && set +a
export READER_BASE_URL=http://localhost:8023/v1 READER_MODEL=Qwen/Qwen3.5-9B
export LME_CONTROLLER_BASE_URL=http://localhost:8023/v1 LME_CONTROLLER_MODEL=Qwen/Qwen3.5-9B
export LME_EMBEDDING_BASE_URL=http://localhost:8114/v1 LME_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
export DATA_ROOT=$E1/data/longmemeval-v2 HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
PY=$E1/miniforge3/envs/lme-v2-release/bin/python
source "$REPO/e1/serve_vllm.sh" || { echo "[arm] vLLM failed to start"; exit 1; }
cd "$REPO"
case "$ARM" in
  gate) "$PY" e1/derive_arm.py --src "$BASE" --dst "$E1/runs/p_small" --premise-check --check-only ${LIMIT:+--limit $LIMIT}
        "$PY" e1/gate_summary.py "$BASE" "$E1/runs/p_small"; rc=$? ;;
  b1|p) DST=$E1/runs/${ARM}_small
        FLAGS="--fix-prompt"; [ "$ARM" = p ] && FLAGS="$FLAGS --premise-check"
        "$PY" e1/derive_arm.py --src "$BASE" --dst "$DST" $FLAGS ${LIMIT:+--limit $LIMIT} || { kill $READER_PID $EMBED_PID; exit 1; }
        rc=0
        for d in web enterprise; do
          "$PY" e1/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
             --data-root "$DATA_ROOT" --output-dir "$DST/${d}_small" \
             --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 ${LIMIT:+--limit $LIMIT} || rc=$?
        done
        [ -z "${LIMIT:-}" ] && "$PY" leaderboard/combine_aggregated_metrics.py \
             "$DST/enterprise_small/aggregated_metrics.json" "$DST/web_small/aggregated_metrics.json" \
             -o "$DST/combined_metrics.json" && cat "$DST/combined_metrics.json" | head -12 ;;
  *) echo "unknown ARM=$ARM"; rc=2 ;;
esac
kill $READER_PID $EMBED_PID 2>/dev/null
exit ${rc:-0}
