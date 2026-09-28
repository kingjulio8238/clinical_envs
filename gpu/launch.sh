#!/bin/bash
# The only way GPU jobs are launched (P9): budget check, pinned profile, detached, server-side timeout required.
#   bash gpu/launch.sh <profile> <projected_usd> gpu/vllm_eval.py --run-name c2-pd --minutes 150 --cmd "..."
# Refuses unless gpu/budget.py says the projection fits under the $25 stop line and the worst case (--minutes at the
# GPU rate) under the $30 credit; refuses a command without --minutes (the Modal function timeout).
set -euo pipefail
P=${1:?profile}; X=${2:?projected dollars}; shift 2
APP=${1:?modal app file}
MIN=""; GPU="H100"; prev=""
for a in "$@"; do
  [[ "$prev" == "--minutes" ]] && MIN=$a
  [[ "$prev" == "--gpu" ]] && GPU=$a
  prev=$a
done
[[ -n "$MIN" ]] || { echo "REFUSED: no --minutes (server-side timeout) in the command" >&2; exit 2; }
cd "$(dirname "$0")/.."
python3 gpu/budget.py check --profile "$P" --projected "$X" --minutes "$MIN" --gpu "$GPU"
echo "profile: $(MODAL_PROFILE=$P modal profile current) | app: $APP | timeout: $MIN min | projected: \$$X"
MODAL_PROFILE=$P modal run --detach "$@"
