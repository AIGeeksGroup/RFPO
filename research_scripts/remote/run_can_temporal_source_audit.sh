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
RUN_NAME="can_step6000_temporal_source_audit_osmesa_seed20260830"
OUTPUT_DIR="$RUNTIME_ROOT/results/$RUN_NAME"
LOG_PATH="$RUNTIME_ROOT/logs/$RUN_NAME.log"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

python audit_temporal_gaussian_source.py \
  --checkpoint "$CHECKPOINT" \
  --output-json "$OUTPUT_DIR/results.json" \
  --seed 20260830 \
  --correlation 0.9 2>&1 | tee "$LOG_PATH"
