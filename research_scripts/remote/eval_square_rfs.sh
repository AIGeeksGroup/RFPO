#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODE="${MODE:-}"
if [[ "$MODE" != "mean" && "$MODE" != "sampled" ]]; then
  echo "Set MODE to mean or sampled" >&2
  exit 1
fi

OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
SQUARE_CHECKPOINT="${SQUARE_CHECKPOINT:-$CHECKPOINT_ROOT/trc7rbt0_step_110000}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-square_h48_rfs_osmesa_seed20260924}"
EXPERIMENT_ID="${EXPERIMENT_ID:-h48}"
RFS_CHECKPOINT="$RUNTIME_ROOT/results/$TRAIN_RUN_NAME/latest.pt"
SEED="${SEED:-20260925}"
RUN_NAME="square_${EXPERIMENT_ID}_candidate_${MODE}_balanced20_seed${SEED}"
OUTPUT_DIR="$RUNTIME_ROOT/results/$RUN_NAME"
LOG_PATH="$RUNTIME_ROOT/logs/$RUN_NAME.log"

if [[ ! -f "$RFS_CHECKPOINT" ]]; then
  echo "Missing RFS checkpoint: $RFS_CHECKPOINT" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite H48 artifact: $RUN_NAME" >&2
  exit 1
fi

DETERMINISTIC=False
if [[ "$MODE" == "mean" ]]; then
  DETERMINISTIC=True
fi

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh
python train_residual_flow_steering.py \
  --mode eval \
  --base-policy-local-path "$SQUARE_CHECKPOINT" \
  --rfs-checkpoint "$RFS_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --task Square \
  --device cuda \
  --seed "$SEED" \
  --num-envs 20 \
  --sampling-steps 10 \
  --n-action-steps 16 \
  --deterministic "$DETERMINISTIC" \
  2>&1 | tee "$LOG_PATH"
