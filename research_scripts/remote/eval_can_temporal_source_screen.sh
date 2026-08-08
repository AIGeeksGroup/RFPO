#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

CHECKPOINT="${CAN_CHECKPOINT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints/95j3noe4_step_6000}"
cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

for sampling_mode in random correlated_random; do
  OUTPUT_DIR="$RUNTIME_ROOT/results/can_step6000_temporal_source_${sampling_mode}_screen_seed20260831"
  python eval_checkpoint.py \
    --local-checkpoint-path "$CHECKPOINT" \
    --load-ema True \
    --eval-env Can \
    --eval-num-episodes 20 \
    --eval-num-envs 20 \
    --balanced-episodes-per-env True \
    --sampling-mode "$sampling_mode" \
    --source-temporal-correlation 0.9 \
    --sampling-steps 10 \
    --seed 20260831 \
    --save-video False \
    --wandb-enable False \
    --output-dir "$OUTPUT_DIR"
done
