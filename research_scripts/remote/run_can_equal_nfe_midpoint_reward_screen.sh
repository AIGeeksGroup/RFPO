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

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
CAN_CHECKPOINT="${CAN_CHECKPOINT:-$CHECKPOINT_ROOT/95j3noe4_step_1000}"
if [[ ! -f "$CAN_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Can checkpoint: $CAN_CHECKPOINT" >&2
  exit 1
fi

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

for condition in control candidate; do
  if [[ "$condition" == "control" ]]; then
    integration_method=euler
    sampling_steps=10
  else
    integration_method=midpoint
    sampling_steps=5
  fi

  for sampling_mode in zero random; do
    run_name="can_step1000_h38b_${condition}_${sampling_mode}_balanced20_seed20260910"
    output_dir="$RUNTIME_ROOT/results/$run_name"
    log_path="$RUNTIME_ROOT/logs/$run_name.log"
    python eval_checkpoint.py \
      --local-checkpoint-path "$CAN_CHECKPOINT" \
      --load-ema True \
      --eval-env Can \
      --eval-num-episodes 20 \
      --eval-num-envs 20 \
      --balanced-episodes-per-env True \
      --sampling-mode "$sampling_mode" \
      --source-prior-mode gaussian \
      --integration-method "$integration_method" \
      --sampling-steps "$sampling_steps" \
      --seed 20260910 \
      --save-video False \
      --wandb-enable False \
      --output-dir "$output_dir" \
      2>&1 | tee "$log_path"
  done
done
