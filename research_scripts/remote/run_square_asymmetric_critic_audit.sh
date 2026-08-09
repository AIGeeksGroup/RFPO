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
SQUARE_CHECKPOINT="${SQUARE_CHECKPOINT:-$CHECKPOINT_ROOT/trc7rbt0_step_110000}"
SEED_START="${SEED_START:-20260926}"
NUM_EPISODES="${NUM_EPISODES:-32}"
NUM_ENVS="${NUM_ENVS:-16}"
RUN_NAME="${RUN_NAME:-square_h50_asymmetric_critic_osmesa_seed${SEED_START}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNTIME_ROOT/results/$RUN_NAME}"
LOG_PATH="${LOG_PATH:-$RUNTIME_ROOT/logs/$RUN_NAME.log}"

if [[ ! -f "$SQUARE_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Square checkpoint: $SQUARE_CHECKPOINT" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite H50 artifact: $RUN_NAME" >&2
  exit 1
fi

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh
python audit_asymmetric_square_critic.py \
  --base-policy-local-path "$SQUARE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --seed-start "$SEED_START" \
  --num-episodes "$NUM_EPISODES" \
  --num-envs "$NUM_ENVS" \
  --sampling-steps 10 \
  --n-action-steps 16 \
  --discount 0.995 \
  --critic-epochs 10 \
  --critic-minibatches 8 \
  --critic-learning-rate 1e-4 \
  2>&1 | tee "$LOG_PATH"
