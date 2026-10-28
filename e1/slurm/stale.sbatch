#!/bin/bash
#SBATCH -p gpu -c 8 --mem=64G --time=02:50:00
#SBATCH -J stale -o /home/w/wang41/e1/stalework/logs/%x.%j.log
#SBATCH --signal=B:TERM@120
# xgpj0 is aarch64; our conda envs are x86_64
#SBATCH --exclude=xgpj0
# STALE port of mem-test v2 (e1/stale/). No self-resubmit; every stage is resumable, so a manual rerun continues.
# Submit with an explicit --gpus=<type>:1, e.g. sbatch --gpus=h100-47:1 --export=ALL,STEPS="prep retrieve gen judge verify",ARM=v2port e1/slurm/stale.sbatch
#   STEPS    : any of prep retrieve gen judge verify (default: all)
#   SET      : scenario set (t1_69 | t1 | t2 | all), default t1_69
#   ARM      : output folder under ~/e1/stalework/runs (required for gen/judge/verify)
#   GEN_ARGS : extra gen.py flags (--prompt official, --order chrono, --conds C0,O_new_only, --limit N, --thinking)
#   JUDGE_MODES : default "state4 official"; JUDGE_REF: reference arm for paired McNemar
set -uo pipefail
killsrv() { for p in ${READER_PID:-} ${EMBED_PID:-}; do [ -n "$p" ] && [ "$p" != 0 ] && kill "$p" 2>/dev/null; done; return 0; }
trap 'killsrv; exit 1' TERM INT
E1=$HOME/e1; REPO=${REPO:-$E1/LongMemEval-V2}; export E1
STEPS=${STEPS:-prep retrieve gen judge verify}; SET=${SET:-t1_69}; RETRIEVER=${RETRIEVER:-bge}
export HF_HOME=$E1/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OPENAI_API_KEY=EMPTY
export READER_MODEL=Qwen/Qwen3.5-9B LOGDIR=$E1/stalework/logs
# short prompts (10 turns): small context, many parallel sequences
export NO_EMBED=1 READER_MAX_LEN=${READER_MAX_LEN:-32768} READER_MAX_SEQS=${READER_MAX_SEQS:-64}
PY=$E1/miniforge3/envs/lme-v2-release/bin/python   # openai client
VPY=$E1/miniforge3/envs/vllm/bin/python            # torch + transformers (bge retrieval)
cd "$REPO"
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
rc=0
has prep && { "$PY" -I e1/stale/prep.py --sets "$SET" || exit 1; }
has retrieve && { "$VPY" -I e1/stale/retrieve.py --set "$SET" --retriever "$RETRIEVER" || exit 1; }
if has gen || has judge; then
  source "$REPO/e1/slurm/serve_vllm.sh" || { echo "[stale] vLLM failed"; killsrv; exit 1; }
  has gen && { "$PY" -I e1/stale/gen.py --set "$SET" --retriever "$RETRIEVER" --arm "${ARM:?}" ${GEN_ARGS:-} || rc=1; }
  if has judge && [ $rc = 0 ]; then
    for m in ${JUDGE_MODES:-state4 official}; do
      "$PY" -I e1/stale/judge.py --arm "$ARM" --mode "$m" ${JUDGE_REF:+--ref $JUDGE_REF} || rc=1
    done
  fi
  killsrv
fi
has verify && { "$PY" -I e1/stale/verify_port.py --arm "$ARM" || rc=1; }
echo "[stale] done rc=$rc"
exit $rc
