#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

RESULT_DIR="$RUNTIME_ROOT/results/spot_h74_antithetic_cross_task"
MANIFEST="$RESULT_DIR/training_seed42.json"
POLL_SECONDS="${POLL_SECONDS:-30}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-10800}"
ORCHESTRATOR_LOG_DIR="$RUNTIME_ROOT/logs/spot_h74_eval_cells"
mkdir -p "$ORCHESTRATOR_LOG_DIR"

elapsed=0
until [[ -s "$MANIFEST" ]]; do
  if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
    echo "Timed out waiting for $MANIFEST" >&2
    exit 1
  fi
  echo "Waiting for H74 training manifest ($elapsed/$WAIT_TIMEOUT_SECONDS seconds)"
  sleep "$POLL_SECONDS"
  ((elapsed += POLL_SECONDS))
done

stages=(zero64 zero32 random64 iid_pair32 antithetic32)
gpus=(1 2 3 4 5)
pids=()

for index in "${!stages[@]}"; do
  stage="${stages[$index]}"
  gpu="${gpus[$index]}"
  output="$RESULT_DIR/${stage}.json"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite $output" >&2
    exit 1
  fi
  echo "Launching $stage on physical GPU $gpu"
  FPO_GPU="$gpu" STAGE="$stage" \
    bash "$SCRIPT_DIR/run_spot_h74_eval.sh" \
    >"$ORCHESTRATOR_LOG_DIR/${stage}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "${stages[$index]} failed; see $ORCHESTRATOR_LOG_DIR/${stages[$index]}.log" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi

echo "All H74 evaluation cells completed; running locked analysis"
FPO_GPU=6 STAGE=analyze bash "$SCRIPT_DIR/run_spot_h74_eval.sh"
