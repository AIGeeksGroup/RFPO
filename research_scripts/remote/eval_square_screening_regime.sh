#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

CHECKPOINT="${SQUARE_CHECKPOINT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints/trc7rbt0_step_110000}"
if [[ ! -f "$CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Square checkpoint: $CHECKPOINT" >&2
  exit 1
fi

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

MODE="${MODE:-zero}"
case "$MODE" in
  zero) ZERO_SAMPLING=True ;;
  random) ZERO_SAMPLING=False ;;
  *) echo "MODE must be zero or random" >&2; exit 2 ;;
esac

RUN_NAME="square_base_${MODE}_balanced20_seed20260912"
python eval_checkpoint.py \
  --local-checkpoint-path "$CHECKPOINT" \
  --load-ema True \
  --eval-env Square \
  --eval-num-episodes 20 \
  --eval-num-envs 20 \
  --balanced-episodes-per-env True \
  --seed 20260912 \
  --sampling-steps 10 \
  --zero-sampling "$ZERO_SAMPLING" \
  --save-video False \
  --wandb-enable False \
  --output-dir "$RUNTIME_ROOT/results/$RUN_NAME" \
  2>&1 | tee "$RUNTIME_ROOT/logs/$RUN_NAME.log"

