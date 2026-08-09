#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODE="${MODE:-}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
case "$MODE" in
  smoke)
    SAMPLING_MODE=curvature_best_of_two
    NUM_EPISODES=8
    NUM_ENVS=8
    SEED=20260921
    MIN_AUDIT_RECORDS=32
    ;;
  control)
    SAMPLING_MODE=random
    NUM_EPISODES=20
    NUM_ENVS=20
    SEED=20260922
    MIN_AUDIT_RECORDS=0
    ;;
  candidate)
    SAMPLING_MODE=curvature_best_of_two
    NUM_EPISODES=20
    NUM_ENVS=20
    SEED=20260922
    MIN_AUDIT_RECORDS=32
    ;;
  *) echo "Set MODE to smoke, control, or candidate" >&2; exit 1 ;;
esac

CHECKPOINT="${CHECKPOINT:-$RUNTIME_ROOT/results/square_h43_control_osmesa_seed20260916/checkpoints/step_25600}"
OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
if [[ ! -f "$OSMESA_LIB/libOSMesa.so.8" ]]; then
  echo "Missing OSMesa runtime: $OSMESA_LIB/libOSMesa.so.8" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

RUN_NAME="square_h46_${MODE}_osmesa_seed${SEED}${RUN_SUFFIX}"
OUTPUT_DIR="$RUNTIME_ROOT/results/$RUN_NAME"
LOG_PATH="$RUNTIME_ROOT/logs/$RUN_NAME.log"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite existing H46 artifact: $RUN_NAME" >&2
  exit 1
fi

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

python eval_checkpoint.py \
  --local-checkpoint-path "$CHECKPOINT" \
  --load-ema False \
  --eval-env Square \
  --eval-num-episodes "$NUM_EPISODES" \
  --eval-num-envs "$NUM_ENVS" \
  --balanced-episodes-per-env True \
  --seed "$SEED" \
  --sampling-mode "$SAMPLING_MODE" \
  --curvature-min-audit-records "$MIN_AUDIT_RECORDS" \
  --curvature-require-both-branches True \
  --sampling-steps 10 \
  --action-steps 16 \
  --save-video False \
  --wandb-enable False \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_PATH"
