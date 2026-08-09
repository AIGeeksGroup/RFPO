#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODE="${MODE:-}"
case "$MODE" in
  mean) ZERO_SAMPLING=True ;;
  sampled) ZERO_SAMPLING=False ;;
  *) echo "Set MODE to mean or sampled" >&2; exit 1 ;;
esac

OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
SQUARE_CHECKPOINT="${SQUARE_CHECKPOINT:-$CHECKPOINT_ROOT/trc7rbt0_step_110000}"
SEED="${SEED:-20260925}"
RUN_NAME="square_h48_control_${MODE}_balanced20_seed${SEED}"
OUTPUT_DIR="$RUNTIME_ROOT/results/$RUN_NAME"
LOG_PATH="$RUNTIME_ROOT/logs/$RUN_NAME.log"

if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite H48 control artifact: $RUN_NAME" >&2
  exit 1
fi

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh
python eval_checkpoint.py \
  --local-checkpoint-path "$SQUARE_CHECKPOINT" \
  --load-ema True \
  --eval-env Square \
  --eval-num-episodes 20 \
  --eval-num-envs 20 \
  --balanced-episodes-per-env True \
  --seed "$SEED" \
  --sampling-steps 10 \
  --action-steps 16 \
  --zero-sampling "$ZERO_SAMPLING" \
  --save-video False \
  --wandb-enable False \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_PATH"
