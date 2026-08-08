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
RUN_NAME="${RUN_NAME:-can_step6000_gmom_microbatch_gradient_audit_osmesa_seed20260908}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNTIME_ROOT/results/$RUN_NAME}"
LOG_PATH="${LOG_PATH:-$RUNTIME_ROOT/logs/$RUN_NAME.log}"

export CAN_CHECKPOINT="${CAN_CHECKPOINT:-$CHECKPOINT_ROOT/95j3noe4_step_6000}"
export NUM_ENVS=16
export DATA_COLLECTION_STEPS=320
export TOTAL_TIMESTEPS=10240
export EVAL_EPISODES=1
export N_ACTION_SAMPLES=8
export LEARNING_RATE_ACTOR=1e-5
export ROLLOUT_FREQ=None
export GAE_LAMBDA=0.99
export MEDIAN_MICROBATCH_GRADIENT_AUDIT=True
export MEDIAN_MICROBATCH_GRADIENT_AUDIT_ITERATION=2
export MEDIAN_MICROBATCH_GRADIENT_AUDIT_CHUNKS=192
export MEDIAN_MICROBATCH_GRADIENT_METHOD=geometric
export MEDIAN_MICROBATCH_GRADIENT_AUDIT_OUTPUT_JSON="$OUTPUT_DIR/gmom_microbatch_gradient_audit.json"
export SEED=20260908
export RUN_NAME OUTPUT_DIR

"$SCRIPT_DIR/run_can_finetune_smoke.sh" 2>&1 | tee "$LOG_PATH"
