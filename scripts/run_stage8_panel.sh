#!/bin/bash
# Stage 8 protocol runs (EVAL_PROTOCOL.md). Two queues in parallel: OpenRouter (the RL candidate, its no-tools
# ablation, a cheap hosted reference) and OpenAI (the GPT-6 Sol anchor). Per-run --max-usd caps; the OpenRouter
# queue also stops when the live account balance falls under $MIN_BALANCE.
#   bash scripts/run_stage8_panel.sh <logdir>
set -u
cd "$(dirname "$0")/.."
LOG=${1:-results/logs}; mkdir -p "$LOG"
set -a; . ./.env; set +a
ALL=patient_diagnosis,evidence_retrieval,context_summarization,specialty_involved,specialty_absent,imaging_indication,differential_diagnosis,test_selection,error_detection,lab_triage,atypical_diagnosis
MIN_BALANCE=${MIN_BALANCE:-1.00}
PY=".venv/bin/python -u -m eval.protocol_run"

balance() { curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/credits \
  | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(round(d['total_credits']-d['total_usage'],3))"; }

openrouter_queue() {
  for spec in "qwen3.5-9b agent $ALL 120 0.45 4096" \
              "qwen3.5-9b single patient_diagnosis,test_selection,differential_diagnosis 120 0.15 4096" \
              "glm-5.3-flash agent $ALL 120 0.20 4096"; do
    set -- $spec
    for task in ${3//,/ }; do
      b=$(balance)
      if python3 -c "import sys; sys.exit(0 if float('$b') < $MIN_BALANCE else 1)"; then
        echo "[$(date +%H:%M:%S)] OPENROUTER STOP: balance \$$b < \$$MIN_BALANCE"; return
      fi
      echo "[$(date +%H:%M:%S)] START $1 $2 $task (balance \$$b)"
      $PY --model "$1" --arm "$2" --task "$task" --n "$4" --workers 6 --max-usd "$5" --max-output-tokens "$6" --out results 2>&1 \
        | grep -E '^\[|^\{"run"|Traceback|Error'
    done
  done
  echo "[$(date +%H:%M:%S)] OPENROUTER QUEUE DONE (balance \$$(balance))"
}

openai_queue() {
  for task in ${ALL//,/ }; do
    echo "[$(date +%H:%M:%S)] START gpt-6-sol agent $task"
    $PY --model gpt-6-sol --arm agent --task "$task" --n 40 --workers 6 --max-usd 1.00 --out results 2>&1 \
      | grep -E '^\[|^\{"run"|Traceback|Error'
  done
  echo "[$(date +%H:%M:%S)] OPENAI QUEUE DONE"
}

openrouter_queue > "$LOG/openrouter.log" 2>&1 &
openai_queue > "$LOG/openai.log" 2>&1 &
wait
echo "[$(date +%H:%M:%S)] ALL QUEUES DONE"
