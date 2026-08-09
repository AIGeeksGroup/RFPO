#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CONDITION="${CONDITION:-}"
case "$CONDITION" in
  control) REFLOW_REGULARIZATION_COEFFICIENT=0.0 ;;
  candidate) REFLOW_REGULARIZATION_COEFFICIENT=0.04 ;;
  *) echo "Set CONDITION to control or candidate" >&2; exit 2 ;;
esac

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export FPO_RENDER_GPU=""
export CAN_CHECKPOINT="${CAN_CHECKPOINT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints/95j3noe4_step_6000}"
export NUM_ENVS=16
export DATA_COLLECTION_STEPS=320
export TOTAL_TIMESTEPS=10240
export EVAL_EPISODES=1
export ROLLOUT_FREQ=None
export N_ACTION_SAMPLES=8
export SEED=20261061
export MASTER_PORT="${MASTER_PORT:-29500}"
export RUN_NAME="can_h61_refpo_${CONDITION}_osmesa_seed${SEED}"
export OUTPUT_DIR="$RUNTIME_ROOT/results/$RUN_NAME"
export REFLOW_REGULARIZATION_COEFFICIENT
export REFPO_FIXED_BATCH_AUDIT=True
export REFPO_FIXED_BATCH_AUDIT_ITERATION=2
export REFPO_FIXED_BATCH_AUDIT_CHUNKS=64
export REFPO_FIXED_BATCH_AUDIT_OUTPUT_JSON="$RUNTIME_ROOT/results/refpo_fixed_batch_audit/${CONDITION}.json"

exec "$SCRIPT_DIR/run_can_finetune_smoke.sh"
