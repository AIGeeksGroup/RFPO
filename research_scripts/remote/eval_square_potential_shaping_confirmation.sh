#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CONDITION="${CONDITION:-}"
MODE="${MODE:-}"
if [[ "$CONDITION" != "control" && "$CONDITION" != "candidate" ]]; then
  echo "Set CONDITION to control or candidate" >&2
  exit 1
fi
case "$MODE" in
  zero) ZERO_SAMPLING=True ;;
  random) ZERO_SAMPLING=False ;;
  *) echo "Set MODE to zero or random" >&2; exit 1 ;;
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

TRAIN_SEED=20260977
TRAIN_RUN_NAME="square_h51_confirmation_${CONDITION}_osmesa_seed${TRAIN_SEED}"
CHECKPOINT="$RUNTIME_ROOT/results/$TRAIN_RUN_NAME/checkpoints/step_25600"
if [[ ! -f "$CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing H51 confirmation checkpoint: $CHECKPOINT" >&2
  exit 1
fi

SEED=20260978
RUN_NAME="square_h51_confirmation_${CONDITION}_${MODE}_balanced50_seed${SEED}"
OUTPUT_DIR="$RUNTIME_ROOT/results/$RUN_NAME"
LOG_PATH="$RUNTIME_ROOT/logs/$RUN_NAME.log"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite existing H51 confirmation artifact: $RUN_NAME" >&2
  exit 1
fi

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh
python eval_checkpoint.py \
  --local-checkpoint-path "$CHECKPOINT" \
  --load-ema False \
  --eval-env Square \
  --eval-num-episodes 50 \
  --eval-num-envs 25 \
  --balanced-episodes-per-env True \
  --seed "$SEED" \
  --sampling-steps 10 \
  --action-steps 16 \
  --zero-sampling "$ZERO_SAMPLING" \
  --save-video False \
  --wandb-enable False \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_PATH"
