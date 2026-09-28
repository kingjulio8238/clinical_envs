#!/bin/bash
# P5: the private-split runs (criterion 7) on this machine against the Modal endpoint of gpu/private_serve.py.
#   bash scripts/run_private.sh <endpoint-url> <token>
# The client and the scorer run here with private/labels_v1.3.db; results in results/local/private/.
set -euo pipefail
URL=${1:?endpoint url}; TOKEN=${2:?bearer token}
cd "$(dirname "$0")/.."
[[ -f private/labels_v1.3.db ]] || { echo "no private overlay on this machine" >&2; exit 2; }
export SH_VLLM_URL="${URL%/}/v1" SH_VLLM_API_KEY="$TOKEN"
# the adapter must be loaded (and differ from the base) before anything is scored under the trained name
python3 - <<PY
import json, os, urllib.request
req = urllib.request.Request(os.environ["SH_VLLM_URL"] + "/models", headers={"Authorization": "Bearer " + os.environ["SH_VLLM_API_KEY"]})
ids = [m["id"] for m in json.load(urllib.request.urlopen(req, timeout=60))["data"]]
assert "Qwen/Qwen3.5-9B" in ids and "qwen3.5-9b-rl" in ids, ids
lp = {}
for name in ("Qwen/Qwen3.5-9B", "qwen3.5-9b-rl"):
    body = json.dumps({"model": name, "max_tokens": 16, "temperature": 0, "logprobs": True, "seed": 0,
                       "messages": [{"role": "user", "content": "Fever, neck stiffness and photophobia for one day. Most likely diagnosis?"}]}).encode()
    r = urllib.request.Request(os.environ["SH_VLLM_URL"] + "/chat/completions", data=body, headers={
        "Authorization": "Bearer " + os.environ["SH_VLLM_API_KEY"], "Content-Type": "application/json"})
    lp[name] = [t["logprob"] for t in json.load(urllib.request.urlopen(r, timeout=600))["choices"][0]["logprobs"]["content"]]
a, b = lp.values()
gap = sum(abs(x - y) for x, y in zip(a, b)) / max(min(len(a), len(b)), 1)
assert gap > 0, "the adapter does not change the base's outputs: refusing to score the base twice"
print("served:", ids, "adapter logprob gap", round(gap, 4))
PY
UNITS=patient_diagnosis,differential_diagnosis,evidence_retrieval,test_selection,atypical_diagnosis
for M in qwen3.5-9b-local qwen3.5-9b-rl; do
  SH_VLLM_MODEL=qwen3.5-9b-rl .venv/bin/python -u -m eval.protocol_run --model $M --local --split private --tasks $UNITS \
    --n 100000 --workers 64 --retry-errors --out results/local/private
done
echo "done: results/local/private (score with scripts/rl_before_after.py --private results/local/private)"
