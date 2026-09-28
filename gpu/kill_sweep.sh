#!/bin/bash
# Stop every stage-C Modal app of a profile (first action after any disconnect, crash or interrupt).
#   bash gpu/kill_sweep.sh <profile>
set -u
P=${1:?profile}
for app in clinical-envs-vllm-eval clinical-envs-train; do
  MODAL_PROFILE=$P modal app stop "$app" -y 2>&1 | tail -1 || true
done
MODAL_PROFILE=$P modal app list 2>&1 | grep -i clinical || echo "no clinical-envs apps running on $P"
