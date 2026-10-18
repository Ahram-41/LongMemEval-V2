#!/bin/bash
#SBATCH -p gpu-long --gpus=h100-96:1 -c 16 --mem=96G --time=12:00:00
#SBATCH -J e1-arm -o %x.%j.log
#SBATCH --signal=B:TERM@180
# Arms derived from the AgentRunbook-R baseline (retrieval reused byte-for-byte).
#   ARM=gate : premise checker only, on all questions -> gate_summary (no reader)
#   ARM=b1   : official system prompt with the backspace fixed; reader rerun
#   ARM=p    : b1 + premise-check note; reader rerun (reuses gate verdicts if present)
#   ARM=b0r  : baseline rows unchanged, reader rerun -> run-to-run noise of the reader
#   ARM=bud  : B1 prompt, only questions that hit the 20k cap in B0, rerun with a 40k budget (diagnostic)
#   ARM=pa / ps : P with note rule any / strict, reader rerun ONLY on noted questions
#                 (unnoted prompts are identical to B1; merge with B1 outputs at analysis time)
# Reader-only: the embedding server is not started. Reader outputs are cached per question, so a
# job killed at its time limit resubmits itself (RESUBMIT_MAX, default 4) and resumes.
set -uo pipefail
E1=$HOME/e1; REPO=$E1/LongMemEval-V2; export E1
ARM=${ARM:?set ARM=gate|b1|p}; BASE=$E1/runs/agentrunbook_r_small
[ -f "$E1/.env" ] && set -a && source "$E1/.env" && set +a
export READER_MODEL=Qwen/Qwen3.5-9B
export LME_CONTROLLER_MODEL=Qwen/Qwen3.5-9B
export LME_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
export DATA_ROOT=$E1/data/longmemeval-v2 HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
PY=$E1/miniforge3/envs/lme-v2-release/bin/python
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}; RESUBMIT_MAX=${RESUBMIT_MAX:-4}
resubmit() {
  kill ${READER_PID:-0} ${EMBED_PID:-0} 2>/dev/null
  if [ "$RESUBMIT_COUNT" -ge "$RESUBMIT_MAX" ]; then echo "[arm] resubmit cap reached"; exit 1; fi
  echo "[arm] time limit: resubmitting ($((RESUBMIT_COUNT+1))/$RESUBMIT_MAX)"
  cd "$E1" && RESUBMIT_COUNT=$((RESUBMIT_COUNT+1)) sbatch --export=ALL -p "$SLURM_JOB_PARTITION" --gpus="$SBATCH_GPUS_RESUB" --time="$SBATCH_TIME_RESUB" -J "$SLURM_JOB_NAME" "$REPO/e1/e1_arm.sbatch"; exit 0
}
trap resubmit TERM
export NO_EMBED=1
source "$REPO/e1/serve_vllm.sh" || { echo "[arm] vLLM failed to start"; exit 1; }
cd "$REPO"
case "$ARM" in
  gate) "$PY" e1/derive_arm.py --src "$BASE" --dst "$E1/runs/p_small" --premise-check --check-only --latency-sample ${LAT_SAMPLE:-10} ${LIMIT:+--limit $LIMIT}
        "$PY" e1/gate_summary.py "$BASE" "$E1/runs/p_small"; rc=$? ;;
  b1|p|b0r) DST=$E1/runs/${ARM}_small
        FLAGS="--fix-prompt"; [ "$ARM" = p ] && FLAGS="$FLAGS --premise-check"; [ "$ARM" = b0r ] && FLAGS=""
        if [ ! -s "$DST/enterprise_small/prompt_rows.jsonl" ]; then
          "$PY" e1/derive_arm.py --src "$BASE" --dst "$DST" $FLAGS ${LIMIT:+--limit $LIMIT} &
          wait $! || { kill $READER_PID 2>/dev/null; exit 1; }
        fi
        rc=0
        for d in web enterprise; do
          "$PY" e1/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
             --data-root "$DATA_ROOT" --output-dir "$DST/${d}_small" \
             --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 ${LIMIT:+--limit $LIMIT} &
          wait $! || rc=$?
        done
        [ -z "${LIMIT:-}" ] && "$PY" leaderboard/combine_aggregated_metrics.py \
             "$DST/enterprise_small/aggregated_metrics.json" "$DST/web_small/aggregated_metrics.json" \
             -o "$DST/combined_metrics.json" && cat "$DST/combined_metrics.json" | head -12 ;;
  pa|ps) RULE=any; [ "$ARM" = ps ] && RULE=strict
        DST=$E1/runs/${ARM}_small; rc=0
        mkdir -p "$DST"; for d in web enterprise; do mkdir -p "$DST/${d}_small"; cp -n "$E1/runs/p_small/${d}_small/premise_check.jsonl" "$DST/${d}_small/" 2>/dev/null; done
        if [ ! -s "$DST/web_small/noted_ids.txt" ]; then
          "$PY" e1/derive_arm.py --src "$BASE" --dst "$DST" --fix-prompt --premise-check --note-rule $RULE --only-noted &
          wait $! || { kill $READER_PID 2>/dev/null; exit 1; }
        fi
        for d in web enterprise; do
          ids=$(cat "$DST/${d}_small/noted_ids.txt" 2>/dev/null); [ -z "$ids" ] && continue
          "$PY" e1/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
             --data-root "$DATA_ROOT" --output-dir "$DST/${d}_small" --question-ids "$ids" \
             --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 &
          wait $! || rc=$?
        done ;;
  bud)  DST=$E1/runs/bud_small; rc=0
        if [ ! -s "$DST/web_small/noted_ids.txt" ]; then
          "$PY" e1/derive_arm.py --src "$BASE" --dst "$DST" --fix-prompt --ids-from-capped &
          wait $! || { kill $READER_PID 2>/dev/null; exit 1; }
        fi
        for d in web enterprise; do
          ids=$(cat "$DST/${d}_small/noted_ids.txt" 2>/dev/null); [ -z "$ids" ] && continue
          "$PY" e1/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
             --data-root "$DATA_ROOT" --output-dir "$DST/${d}_small" --question-ids "$ids" --max-completion-tokens 40000 \
             --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 &
          wait $! || rc=$?
        done ;;
  *) echo "unknown ARM=$ARM"; rc=2 ;;
esac
kill $READER_PID $EMBED_PID 2>/dev/null
exit ${rc:-0}
