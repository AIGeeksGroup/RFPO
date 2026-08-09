#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
SQUARE_CHECKPOINT="${SQUARE_CHECKPOINT:-$CHECKPOINT_ROOT/trc7rbt0_step_110000}"
SEED="${SEED:-20260924}"
COLLECTION_STEPS="${COLLECTION_STEPS:-320}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-25600}"
RATIO_MODE="${RATIO_MODE:-joint}"
RUN_NAME="${RUN_NAME:-square_h48_rfs_osmesa_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNTIME_ROOT/results/$RUN_NAME}"
LOG_PATH="${LOG_PATH:-$RUNTIME_ROOT/logs/$RUN_NAME.log}"

if [[ ! -f "$SQUARE_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Square checkpoint: $SQUARE_CHECKPOINT" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite H48 artifact: $RUN_NAME" >&2
  exit 1
fi

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh
python train_residual_flow_steering.py \
  --mode train \
  --base-policy-local-path "$SQUARE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --task Square \
  --device cuda \
  --seed "$SEED" \
  --num-envs 16 \
  --sampling-steps 10 \
  --n-action-steps 16 \
  --collection-steps "$COLLECTION_STEPS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --ratio-mode "$RATIO_MODE" \
  2>&1 | tee "$LOG_PATH"
