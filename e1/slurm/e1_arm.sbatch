#!/bin/bash
#SBATCH -p gpu-long --gpus=h100-96:1 -c 16 --mem=96G --time=12:00:00
#SBATCH -J e1-arm -o %x.%j.log
#SBATCH --signal=B:TERM@180
# xgpj0 is aarch64; our conda envs are x86_64
#SBATCH --exclude=xgpj0
# Arms derived from the AgentRunbook-R baseline (retrieval reused byte-for-byte).
#   ARM=gate : premise checker only, on all questions -> gate_summary (no reader)
#   ARM=b1   : official system prompt with the backspace fixed; reader rerun
#   ARM=p    : b1 + premise-check note; reader rerun (reuses gate verdicts if present)
#   ARM=b0r  : baseline rows unchanged, reader rerun -> run-to-run noise of the reader
#   ARM=mid  : mid-thinking injection (mi = verdict, mn = verdict-free nudge) on P-any noted questions
#   ARM=bud  : B1 prompt, only questions that hit the 20k cap in B0, rerun with a 40k budget (diagnostic)
#   ARM=form : E2 S1c memory form (FORM=noraw|rawonly) on B1 contexts, all questions
#   ARM=e2   : E2 retrieval arm (E2ARM, E2CTX, E2RET); reader rerun only on questions whose raw blocks changed
#   ARM=pa / ps : P with note rule any / strict, reader rerun ONLY on noted questions
#                 (unnoted prompts are identical to B1; merge with B1 outputs at analysis time)
#   ARM=think: rerun_with_thinking.py (error analysis: reasoning_content on a fixed id list)
# Reader-only: the embedding server is not started. Reader outputs are cached per question, so a
# job killed at its time limit resubmits itself (RESUBMIT_MAX, default 4) and resumes.
set -uo pipefail
# never "kill 0": that signals the whole process group, including this script
killsrv() { for p in ${READER_PID:-} ${EMBED_PID:-}; do [ -n "$p" ] && [ "$p" != 0 ] && kill "$p" 2>/dev/null; done; return 0; }
E1=$HOME/e1; REPO=$E1/LongMemEval-V2; export E1
ARM=${ARM:?set ARM=gate|b1|p|...|e2}; BASE=$E1/runs/agentrunbook_r_small
[ -f "$E1/.env" ] && set -a && source "$E1/.env" && set +a
# NO_JUDGE=1: never call the paid judge from the harness (scoring then fails fast after outputs are cached)
[ -n "${NO_JUDGE:-}" ] && export OPENAI_API_KEY=EMPTY
export READER_MODEL=Qwen/Qwen3.5-9B
export LME_CONTROLLER_MODEL=Qwen/Qwen3.5-9B
export LME_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
export DATA_ROOT=$E1/data/longmemeval-v2 HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
PY=$E1/miniforge3/envs/lme-v2-release/bin/python
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}; RESUBMIT_MAX=${RESUBMIT_MAX:-4}
resubmit() {
  killsrv
  # cancel guard: touch $E1/STOP_RESUBMIT (or $E1/STOP_<jobname>) before scancel to prevent the self-resubmit
  if [ -f "$E1/STOP_RESUBMIT" ] || [ -f "$E1/STOP_${SLURM_JOB_NAME:-x}" ]; then echo "[arm] STOP file present: not resubmitting"; exit 0; fi
  if [ "$RESUBMIT_COUNT" -ge "$RESUBMIT_MAX" ]; then echo "[arm] resubmit cap reached"; exit 1; fi
  echo "[arm] time limit: resubmitting ($((RESUBMIT_COUNT+1))/$RESUBMIT_MAX)"
  cd "$E1" && RESUBMIT_COUNT=$((RESUBMIT_COUNT+1)) sbatch --export=ALL -p "$SLURM_JOB_PARTITION" --gpus="$SBATCH_GPUS_RESUB" --time="$SBATCH_TIME_RESUB" -J "$SLURM_JOB_NAME" "$REPO/e1/slurm/e1_arm.sbatch"; exit 0
}
trap resubmit TERM
export NO_EMBED=1
# local judge: short prompts, many parallel requests
[ "$ARM" = ljudge ] && export READER_MAX_LEN=${READER_MAX_LEN:-65536} READER_MAX_SEQS=${READER_MAX_SEQS:-32}
cd "$REPO"
# Pre-serve steps (2026-11-02): the reranker needs the GPU to itself; context builds need more RAM than the login node allows.
#   ARM=rr : Qwen3-Reranker-8B over the E2 candidate pools (rerank_score.py, vllm env), select, build contexts, then run as ARM=e2.
#   E2BUILD=1 : build <dom>_contexts.jsonl from E2RET before the reader starts.  E2IDS=<file> -> E2FLAGS="--ids-file <file>".
if [ "$ARM" = rr ]; then
  need=; for d in web enterprise; do [ -s "$E1/e2work/rerank/scores_$d.jsonl" ] || need=1; done
  if [ -n "$need" ]; then
    PATH="$E1/miniforge3/envs/vllm/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} \
      "$E1/miniforge3/envs/vllm/bin/python" e1/oneoff/rerank_score.py --gpu-frac ${RR_GPU_FRAC:-0.85} || { echo "[arm] rerank_score failed"; exit 1; }
  fi
  "$PY" -I e1/oneoff/rerank_arm.py select || exit 1
  E2BUILD=1; E2ARM=${E2ARM:-rr_small}; E2CTX=$E1/e2work/contexts/rr; E2RET=$E1/e2work/retrieved/rr; E2IDS=${E2IDS:-$E1/e2work/rerank/ids.txt}; ARM=e2
fi
[ -n "${E2IDS:-}" ] && E2FLAGS="${E2FLAGS:-} --ids-file $E2IDS"
if [ -n "${E2BUILD:-}" ]; then
  for d in web enterprise; do
    [ -s "$E2CTX/${d}_contexts.jsonl" ] || "$PY" -I e1/pipeline/build_context.py --domain $d --retrieved "$E2RET/$d.jsonl" --out "$E2CTX/${d}_contexts.jsonl" || { echo "[arm] build_context $d failed"; exit 1; }
  done
fi
source "$REPO/e1/slurm/serve_vllm.sh" || { echo "[arm] vLLM failed to start"; exit 1; }
cd "$REPO"
case "$ARM" in
  gate) "$PY" e1/pipeline/derive_arm.py --src "$BASE" --dst "$E1/runs/p_small" --premise-check --check-only --latency-sample ${LAT_SAMPLE:-10} ${LIMIT:+--limit $LIMIT}
        "$PY" e1/analysis/gate_summary.py "$BASE" "$E1/runs/p_small"; rc=$? ;;
  b1|p|b0r) DST=$E1/runs/${ARM}_small
        FLAGS="--fix-prompt"; [ "$ARM" = p ] && FLAGS="$FLAGS --premise-check"; [ "$ARM" = b0r ] && FLAGS=""
        if [ ! -s "$DST/enterprise_small/prompt_rows.jsonl" ]; then
          "$PY" e1/pipeline/derive_arm.py --src "$BASE" --dst "$DST" $FLAGS ${LIMIT:+--limit $LIMIT} &
          wait $! || { killsrv; exit 1; }
        fi
        rc=0
        for d in web enterprise; do
          "$PY" e1/pipeline/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
             --data-root "$DATA_ROOT" --output-dir "$DST/${d}_small" \
             --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 ${LIMIT:+--limit $LIMIT} &
          wait $! || rc=$?
        done
        [ -z "${LIMIT:-}" ] && "$PY" leaderboard/combine_aggregated_metrics.py \
             "$DST/enterprise_small/aggregated_metrics.json" "$DST/web_small/aggregated_metrics.json" \
             -o "$DST/combined_metrics.json" && cat "$DST/combined_metrics.json" | head -12 ;;
  pa|ps|pnf) RULE=any; [ "$ARM" = ps ] && RULE=strict; [ "$ARM" = pnf ] && RULE=nfq
        DST=$E1/runs/${ARM}_small; rc=0
        mkdir -p "$DST"; for d in web enterprise; do mkdir -p "$DST/${d}_small"; cp -n "$E1/runs/p_small/${d}_small/premise_check.jsonl" "$DST/${d}_small/" 2>/dev/null; done
        if [ ! -s "$DST/web_small/noted_ids.txt" ]; then
          "$PY" e1/pipeline/derive_arm.py --src "$BASE" --dst "$DST" --fix-prompt --premise-check --note-rule $RULE --only-noted &
          wait $! || { killsrv; exit 1; }
        fi
        for d in web enterprise; do
          ids=$(cat "$DST/${d}_small/noted_ids.txt" 2>/dev/null); [ -z "$ids" ] && continue
          "$PY" e1/pipeline/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
             --data-root "$DATA_ROOT" --output-dir "$DST/${d}_small" --question-ids "$ids" \
             --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 &
          wait $! || rc=$?
        done ;;
  bud)  DST=$E1/runs/bud_small; rc=0
        if [ ! -s "$DST/web_small/noted_ids.txt" ]; then
          "$PY" e1/pipeline/derive_arm.py --src "$BASE" --dst "$DST" --fix-prompt --ids-from-capped &
          wait $! || { killsrv; exit 1; }
        fi
        for d in web enterprise; do
          ids=$(cat "$DST/${d}_small/noted_ids.txt" 2>/dev/null); [ -z "$ids" ] && continue
          "$PY" e1/pipeline/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
             --data-root "$DATA_ROOT" --output-dir "$DST/${d}_small" --question-ids "$ids" --max-completion-tokens 40000 \
             --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 &
          wait $! || rc=$?
        done ;;
  mid)  rc=0
        "$PY" e1/oneoff/check_continue.py || true
        for A in ${MID_ARMS:-mi mn}; do
          "$PY" e1/oneoff/mid_inject.py --arm $A --k ${MID_K:-12000} ${LIMIT:+--limit $LIMIT} &
          wait $! || { rc=1; continue; }
          for d in web enterprise; do
            ids=$(cat "$E1/runs/${A}_small/${d}_small/noted_ids.txt" 2>/dev/null); [ -z "$ids" ] && continue
            "$PY" e1/pipeline/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
               --data-root "$DATA_ROOT" --output-dir "$E1/runs/${A}_small/${d}_small" --question-ids "$ids" \
               --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 &
            wait $! || rc=$?
          done
        done ;;
  e2)   # E2 retrieval arm: reader rerun only where raw blocks differ from B1 (or queries were regenerated)
        # needs E2ARM (folder name), E2CTX (dir with <dom>_contexts.jsonl), E2RET (dir with <dom>.jsonl)
        DST=$E1/runs/${E2ARM:?}; rc=0
        if [ ! -s "$DST/enterprise_small/rerun_ids.txt" ]; then
          "$PY" e1/pipeline/make_prompt_rows.py --contexts "${E2CTX:?}" --retrieved "${E2RET:?}" --dst "$DST" ${E2FLAGS:-} &
          wait $! || { killsrv; exit 1; }
        fi
        for d in ${E2DOMAINS:-web enterprise}; do   # E2DOMAINS lets two jobs split one arm by domain
          ids=$(cat "$DST/${d}_small/rerun_ids.txt" 2>/dev/null); [ -z "$ids" ] && continue
          "$PY" e1/pipeline/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
             --data-root "$DATA_ROOT" --output-dir "$DST/${d}_small" --question-ids "$ids" \
             --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 &
          wait $! || rc=$?
        done ;;
  form) # E2 S1c: B1 contexts with only some memory sections (FORM=noraw keeps notes+events, FORM=rawonly keeps raw)
        KEEP=notes,events; [ "${FORM:?}" = rawonly ] && KEEP=raw
        DST=$E1/runs/s1_${FORM}_small; rc=0
        if [ ! -s "$DST/enterprise_small/prompt_rows.jsonl" ]; then
          "$PY" e1/pipeline/derive_arm.py --src "$BASE" --dst "$DST" --fix-prompt --keep-sections $KEEP ${LIMIT:+--limit $LIMIT} &
          wait $! || { killsrv; exit 1; }
        fi
        for d in web enterprise; do
          "$PY" e1/pipeline/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
             --data-root "$DATA_ROOT" --output-dir "$DST/${d}_small" \
             --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 ${LIMIT:+--limit $LIMIT} &
          wait $! || rc=$?
        done ;;
  pfuse) # E2 C: P-any premise note on top of an E2 arm, false-premise questions only (PSRC = source arm folder)
        SRC=$E1/runs/${PSRC:?}; DST=$E1/runs/${PDST:?}; rc=0
        if [ ! -s "$DST/web_small/noted_ids.txt" ]; then
          "$PY" e1/pipeline/derive_arm.py --src "$SRC" --dst "$DST" --premise-check --note-rule any --only-noted --ids-file "${PIDS:?}" &
          wait $! || { killsrv; exit 1; }
        fi
        for d in web enterprise; do
          ids=$(cat "$DST/${d}_small/noted_ids.txt" 2>/dev/null); [ -z "$ids" ] && continue
          "$PY" e1/pipeline/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
             --data-root "$DATA_ROOT" --output-dir "$DST/${d}_small" --question-ids "$ids" \
             --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 &
          wait $! || rc=$?
        done ;;
  ljudge) # local Qwen judge for LLM-judged questions (GPT-5.2 credits exhausted); LJ_ARMS comma list
        "$PY" e1/analysis/local_judge.py --arms "${LJ_ARMS:-b1_small,e2_fusion_small}" ${LJ_IDS:+--ids-file $LJ_IDS} &
        wait $!; rc=$? ;;
  e2phase) # prioritised phases of an E2 arm (split_phases.py): PHARM=<arm>, E2DOMAINS=<domain>, phases p1 p2 p3 in order
        rc=0
        for P in ${PHASES:-p1 p2 p3}; do
          for d in ${E2DOMAINS:-web enterprise}; do
            DST=$E1/runs/${PHARM:?}_${P}_small/${d}_small
            ids=$(cat "$DST/ids.txt" 2>/dev/null); [ -z "$ids" ] && continue
            n=$(tr , "\n" < "$DST/ids.txt" | wc -l); c=$(grep -c question_id "$DST/reader_outputs_cache.jsonl" 2>/dev/null || echo 0)
            echo "[e2phase] $P $d: $c/$n cached"
            "$PY" e1/pipeline/run_e1.py --index-root "$E1/index" --method agentrunbook_r --domain "$d" --tier small \
               --data-root "$DATA_ROOT" --output-dir "$DST" --question-ids "$ids" \
               --reader-max-concurrent-requests 8 --prompt-build-max-workers 8 &
            wait $! || true   # scoring fails by design under NO_JUDGE; outputs are cached
          done
        done ;;
  think) # error analysis 2026-10-26: replay cached prompts, keep reasoning_content (THINK_ARMS, THINK_CONC)
        rc=0
        for A in ${THINK_ARMS:-b1 fusion}; do
          "$PY" e1/oneoff/rerun_with_thinking.py run --arm $A --concurrency ${THINK_CONC:-6} ${LIMIT:+--limit $LIMIT} &
          wait $! || rc=$?
        done
        "$PY" e1/oneoff/rerun_with_thinking.py score || true ;;
  iv)   # reader interventions 2026-10-28 (reader_interventions.py): IV_VARIANT=base|strip; IV_WRAPOLD=1 first
        rc=0
        [ -n "${IV_WRAPOLD:-}" ] && { "$PY" e1/oneoff/reader_interventions.py wrap-old --concurrency 8 & wait $! || true; }
        "$PY" e1/oneoff/reader_interventions.py run --variant "${IV_VARIANT:?}" --concurrency ${IV_CONC:-8} ${LIMIT:+--limit $LIMIT} &
        wait $! || rc=$? ;;
  iv2)  # forced-format wrap-up on all capped rows, then score (after both iv variants finished)
        "$PY" e1/oneoff/reader_interventions.py wrap2 --concurrency 8 & wait $!; rc=$?
        "$PY" e1/oneoff/reader_interventions.py score || true ;;
  ivrep) # repeated sampling on the label-anchor ids: base / strip / neutral labels (IVREP_SAMPLES, IVREP_IDS)
        "$PY" e1/oneoff/reader_interventions.py rep --samples ${IVREP_SAMPLES:-4} --concurrency ${IV_CONC:-12} ${IVREP_IDS:+--ids $IVREP_IDS} & wait $!; rc=$? ;;
  ivtrad) # TRAD-style aligned step records on the B1 prompts (trad_steps.py), forced wrap-up inline; then score.
        # IV_VARIANT=dedup (control: chrome once + exact repeats -> pointers) or trad2 (second trad sample)
        "$PY" e1/oneoff/reader_interventions.py run --variant ${IV_VARIANT:-trad} --concurrency ${IV_CONC:-12} ${LIMIT:+--limit $LIMIT} & wait $!; rc=$?
        [ $rc = 0 ] && { "$PY" e1/oneoff/reader_interventions.py score || true; } ;;
  trc)  # C reader stage (CP5101 trunc-reader-plan-2026-11-02.md): build prompt rows on the GPU node (official
        # counter), check them, then 2 samples per arm with forced wrap-up inline. C1 = A0 vs C on the A0-truncated
        # questions; C2 = RR vs RR+C on the RR-truncated ones. Contexts from trunc_c.py build (CPU).
        V2=$E1/v2work; TR=$V2/trunc; rc=0
        mk() { "$PY" e1/pipeline/make_prompt_rows.py --contexts "$1" --retrieved "$2" --dst "$3" --force-rerun --ids-file "$4" ${5:-} & wait $!; }
        ids() { cat "$1/web_truncated_ids.txt"; echo -n ,; cat "$1/enterprise_truncated_ids.txt"; }
        ids "$TR/head" > "$V2/c1_ids.txt"; ids "$TR/rr_head" > "$V2/c2_ids.txt"
        [ -s "$E1/runs/c1_a0_small/enterprise_small/prompt_rows.jsonl" ] || mk "$E1/e2work/contexts/e2_v" "$E1/e2work/retrieved/e2_v" "$E1/runs/c1_a0_small" "$V2/c1_ids.txt" "--only regenerated" || { killsrv; exit 1; }
        [ -s "$E1/runs/c1_c_small/enterprise_small/prompt_rows.jsonl" ] || mk "$TR/head" "$E1/e2work/retrieved/e2_v" "$E1/runs/c1_c_small" "$V2/c1_ids.txt" "--only regenerated" || { killsrv; exit 1; }
        [ -s "$E1/runs/c2_rc_small/enterprise_small/prompt_rows.jsonl" ] || mk "$TR/rr_head" "$E1/e2work/retrieved/rr" "$E1/runs/c2_rc_small" "$V2/c2_ids.txt" || { killsrv; exit 1; }
        "$PY" e1/oneoff/trunc_c.py rows-check --rows "$E1/runs/c1_a0_small" --same-as e2_ordA0_small,e2_ordA0rest_small || { killsrv; exit 1; }
        "$PY" e1/oneoff/trunc_c.py rows-check --rows "$E1/runs/c1_c_small" --untruncated || { killsrv; exit 1; }
        "$PY" e1/oneoff/trunc_c.py rows-check --rows "$E1/runs/c2_rc_small" --untruncated || { killsrv; exit 1; }
        for spec in ${TRC_ARMS:-c1a0_s1:c1_a0:c1 c1c_s1:c1_c:c1 c2rr_s1:rr:c2 c2rc_s1:c2_rc:c2 c1a0_s2:c1_a0:c1 c1c_s2:c1_c:c1 c2rr_s2:rr:c2 c2rc_s2:c2_rc:c2}; do
          IFS=: read -r V SRCARM SET <<< "$spec"
          "$PY" e1/oneoff/reader_interventions.py run --variant "$V" --src "$E1/runs/${SRCARM}_small" --ids-file "$V2/${SET}_ids.txt" --concurrency ${IV_CONC:-12} & wait $! || rc=$?
        done ;;
  trc3) # 2026-11-05 cut-shape test (CP5101 experiments/trunc-shape-plan-2026-11-05.md): other per-page cut shapes on the
        # 74 RR-truncated rule-scored questions, 2 samples each, forced wrap-up inline; control = the existing per-page head
        # cut samples iv_c2rc_s{1,2}. Contexts from `trunc_c.py build --prefix rr_ --variants <V>` (CPU).
        # TRC3_VARIANTS: any of uniform mid chrome boiler head (one job per variant keeps each job under ~2.5 h);
        # head = runs/c3_rr_head_small -> c2_rc_small (same-night control). TRC3_IDS: question ids file (default c2_ids.txt). TRC3_SAMPLES: default "1,2"; pass via the environment (TRC3_SAMPLES="3 4" sbatch --export=ALL), not inside --export=, which splits on commas
        V2=$E1/v2work; TR=$V2/trunc; rc=0
        for V in ${TRC3_VARIANTS:?}; do
          DST=$E1/runs/c3_rr_${V}_small
          [ -s "$DST/enterprise_small/prompt_rows.jsonl" ] || { "$PY" e1/pipeline/make_prompt_rows.py --contexts "$TR/rr_$V" --retrieved "$E1/e2work/retrieved/rr" --dst "$DST" --force-rerun --ids-file "${TRC3_IDS:-$V2/c2_ids.txt}" & wait $!; } || { killsrv; exit 1; }
          "$PY" e1/oneoff/trunc_c.py rows-check --rows "$DST" --untruncated || { killsrv; exit 1; }
          SMP=${TRC3_SAMPLES:-1,2}; for S in ${SMP//,/ }; do
            "$PY" e1/oneoff/reader_interventions.py run --variant "c3${V}_s$S" --src "$DST" --ids-file "${TRC3_IDS:-$V2/c2_ids.txt}" --concurrency ${IV_CONC:-12} & wait $! || rc=$?
          done
        done ;;
  *) echo "unknown ARM=$ARM"; rc=2 ;;
esac
killsrv
exit ${rc:-0}
