#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
if [[ ! -f "$OSMESA_LIB/libOSMesa.so.8" ]]; then
  echo "Missing OSMesa runtime: $OSMESA_LIB/libOSMesa.so.8" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

for condition in control candidate; do
  train_run="can_step6000_h30_${condition}_osmesa_seed20260828"
  checkpoint="$RUNTIME_ROOT/results/$train_run/checkpoints/latest"
  if [[ ! -f "$checkpoint/policy/model.safetensors" ]]; then
    echo "Missing trained checkpoint: $checkpoint" >&2
    exit 1
  fi

  for sampling_mode in zero random; do
    run_name="can_step6000_h30b_${condition}_${sampling_mode}_balanced20_seed20260906"
    output_dir="$RUNTIME_ROOT/results/$run_name"
    log_path="$RUNTIME_ROOT/logs/$run_name.log"
    python eval_checkpoint.py \
      --local-checkpoint-path "$checkpoint" \
      --load-ema False \
      --eval-env Can \
      --eval-num-episodes 20 \
      --eval-num-envs 20 \
      --balanced-episodes-per-env True \
      --sampling-mode "$sampling_mode" \
      --sampling-steps 10 \
      --seed 20260906 \
      --save-video False \
      --wandb-enable False \
      --output-dir "$output_dir" 2>&1 | tee "$log_path"
  done
done
