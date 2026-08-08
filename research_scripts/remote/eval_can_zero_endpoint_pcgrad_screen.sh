#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CONDITION="${CONDITION:-}"
if [[ "$CONDITION" != "control" && "$CONDITION" != "candidate" ]]; then
  echo "Set CONDITION to control or candidate" >&2
  exit 1
fi

OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

TRAIN_RUN="can_step6000_h30_${CONDITION}_osmesa_seed20260828"
CHECKPOINT="$RUNTIME_ROOT/results/$TRAIN_RUN/checkpoints/latest"
if [[ ! -f "$CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing trained checkpoint: $CHECKPOINT" >&2
  exit 1
fi

for sampling_mode in zero random; do
  OUTPUT_DIR="$RUNTIME_ROOT/results/can_step6000_h30_${CONDITION}_${sampling_mode}_eval_seed20260829"
  python eval_checkpoint.py \
    --local-checkpoint-path "$CHECKPOINT" \
    --load-ema False \
    --eval-env Can \
    --eval-num-episodes 20 \
    --eval-num-envs 50 \
    --sampling-mode "$sampling_mode" \
    --sampling-steps 10 \
    --seed 20260829 \
    --save-video False \
    --wandb-enable False \
    --output-dir "$OUTPUT_DIR"
done
