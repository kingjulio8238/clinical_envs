#!/bin/bash
# Stage 8 protocol runs (EVAL_PROTOCOL.md §4): the RL candidate (Qwen3.5-9B, OpenRouter) on every scoring unit with
# tools, its no-tools ablation on three units, and the GPT-6 Sol anchor (OpenAI direct). Each (model, arm) is one
# process with one interleaved worker pool over its units; each carries a whole-run --max-usd cap, and the
# OpenRouter runs also stop when the live balance falls under $MIN_BALANCE. Completed episodes are kept, so a
# rerun of this script resumes. Optional generator row (unranked): KIMI=1 adds Kimi K2.5 at 20 per unit.
#   bash scripts/run_stage8_panel.sh <logdir> [smoke]
set -u
cd "$(dirname "$0")/.."
LOG=${1:-results/logs}; mkdir -p "$LOG"
MODE=${2:-full}
set -a; . ./.env; set +a
UNITS=patient_diagnosis,evidence_retrieval,context_summarization,specialty_conditioned,imaging_indication,differential_diagnosis,test_selection,error_detection,lab_triage,atypical_diagnosis
ABLATION=patient_diagnosis,test_selection,differential_diagnosis
MIN_BALANCE=${MIN_BALANCE:-1.00}
PY=".venv/bin/python -u -m eval.protocol_run"
if [ "$MODE" = smoke ]; then N_Q=3; N_S=3; N_K=3; SMOKE=--smoke; else N_Q=120; N_S=40; N_K=20; SMOKE=; fi

openrouter() {
  $PY --model qwen3.5-9b --arm agent  --tasks "$UNITS"    --n $N_Q --workers 16 --max-usd ${QWEN_MAX_USD:-4.50} \
      --min-balance "$MIN_BALANCE" --max-output-tokens 4096 --out results $SMOKE
  $PY --model qwen3.5-9b --arm single --tasks "$ABLATION" --n $N_Q --workers 16 --max-usd ${QWEN_SINGLE_MAX_USD:-0.30} \
      --min-balance "$MIN_BALANCE" --max-output-tokens 16384 --out results $SMOKE   # one call: the whole thinking budget
  if [ "${KIMI:-0}" = 1 ]; then
    $PY --model kimi-k2.5 --arm agent --tasks "$UNITS" --n $N_K --workers 8 --max-usd ${KIMI_MAX_USD:-2.00} \
        --min-balance "$MIN_BALANCE" --max-output-tokens 4096 --out results $SMOKE
  fi
  echo "[$(date +%H:%M:%S)] OPENROUTER DONE"
}

openai() {
  $PY --model gpt-6-sol --arm agent --tasks "$UNITS" --n $N_S --workers 8 --max-usd ${SOL_MAX_USD:-12.00} --out results $SMOKE
  echo "[$(date +%H:%M:%S)] OPENAI DONE"
}

openrouter > "$LOG/openrouter.log" 2>&1 &
openai > "$LOG/openai.log" 2>&1 &
wait
echo "[$(date +%H:%M:%S)] ALL DONE"
