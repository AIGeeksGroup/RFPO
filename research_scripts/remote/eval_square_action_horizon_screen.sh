#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODE="${MODE:-}"
ACTION_STEPS="${ACTION_STEPS:-}"
case "$MODE" in
  zero) ZERO_SAMPLING=True ;;
  random) ZERO_SAMPLING=False ;;
  *) echo "Set MODE to zero or random" >&2; exit 1 ;;
esac
case "$ACTION_STEPS" in
  8|16) ;;
  *) echo "Set ACTION_STEPS to 8 or 16" >&2; exit 1 ;;
esac

OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
if [[ ! -f "$OSMESA_LIB/libOSMesa.so.8" ]]; then
  echo "Missing OSMesa runtime: $OSMESA_LIB/libOSMesa.so.8" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

CHECKPOINT="${CHECKPOINT:-$RUNTIME_ROOT/results/square_h41_control_osmesa_seed20260913/checkpoints/latest}"
if [[ ! -f "$CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing frozen H41 control checkpoint: $CHECKPOINT" >&2
  exit 1
fi

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

SEED="${SEED:-20260915}"
RUN_NAME="square_h42_action${ACTION_STEPS}_${MODE}_balanced20_seed${SEED}"
OUTPUT_DIR="$RUNTIME_ROOT/results/$RUN_NAME"
LOG_PATH="$RUNTIME_ROOT/logs/$RUN_NAME.log"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite existing H42 artifact: $RUN_NAME" >&2
  exit 1
fi

python eval_checkpoint.py \
  --local-checkpoint-path "$CHECKPOINT" \
  --load-ema False \
  --eval-env Square \
  --eval-num-episodes 20 \
  --eval-num-envs 20 \
  --balanced-episodes-per-env True \
  --seed "$SEED" \
  --sampling-steps 10 \
  --action-steps "$ACTION_STEPS" \
  --zero-sampling "$ZERO_SAMPLING" \
  --save-video False \
  --wandb-enable False \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_PATH"
