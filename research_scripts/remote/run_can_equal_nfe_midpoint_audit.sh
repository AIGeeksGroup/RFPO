#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
CAN_CHECKPOINT="${CAN_CHECKPOINT:-$CHECKPOINT_ROOT/95j3noe4_step_1000}"
AUDIT_TAG="${AUDIT_TAG:-can_equal_nfe_midpoint_seed20260909}"
OUTPUT_JSON="$RUNTIME_ROOT/results/${AUDIT_TAG}.json"
LOG_PATH="$RUNTIME_ROOT/logs/${AUDIT_TAG}.log"

if [[ ! -f "$CAN_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Can checkpoint: $CAN_CHECKPOINT" >&2
  exit 1
fi

python evaluate_flow_geometry.py \
  --checkpoint-path "$CAN_CHECKPOINT" \
  --output-json "$OUTPUT_JSON" \
  --dataset ankile/robomimic-mh-can-image \
  --sampling-steps 64,10 \
  --reference-steps 64 \
  --batch-size 64 \
  --num-batches 2 \
  --diversity-samples 64 \
  --seed 20260909 \
  --num-workers "${AUDIT_WORKERS:-2}" \
  --device cuda \
  --load-ema \
  --control-steps 10 \
  --candidate-method midpoint \
  --candidate-steps 5 \
  2>&1 | tee "$LOG_PATH"
