#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
SQUARE_CHECKPOINT="${SQUARE_CHECKPOINT:-$CHECKPOINT_ROOT/trc7rbt0_step_110000}"
ARCHIVED_RECORDS="${ARCHIVED_RECORDS:-$PROJECT_ROOT/experiments/group-relative-episode-audit/results/audit/records.json}"
MODE="${MODE:-smoke}"
FPO_GPU="${FPO_GPU:-0}"
FPO_RENDER_GPU="${FPO_RENDER_GPU:-$FPO_GPU}"
RUN_NAME="${RUN_NAME:-square_h64_deterministic_reference_${MODE}_osmesa}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNTIME_ROOT/results/$RUN_NAME}"
LOG_PATH="${LOG_PATH:-$RUNTIME_ROOT/logs/$RUN_NAME.log}"

case "$MODE" in
  smoke|audit) ;;
  *) echo "MODE must be smoke or audit" >&2; exit 1 ;;
esac
if [[ ! -f "$SQUARE_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Square checkpoint: $SQUARE_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$ARCHIVED_RECORDS" ]]; then
  echo "Missing archived H63 records: $ARCHIVED_RECORDS" >&2
  exit 1
fi
if [[ ! -f "$OSMESA_LIB/libOSMesa.so.8" ]]; then
  echo "Missing OSMesa runtime: $OSMESA_LIB/libOSMesa.so.8" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite H64 artifact: $RUN_NAME" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")" "$(dirname "$LOG_PATH")"
cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh
export CUDA_VISIBLE_DEVICES="$FPO_GPU"
export FPO_RENDER_GPU
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONUNBUFFERED=1

python audit_deterministic_reference_advantages.py \
  --local-checkpoint-path "$SQUARE_CHECKPOINT" \
  --archived-records-path "$ARCHIVED_RECORDS" \
  --output-dir "$OUTPUT_DIR" \
  --mode "$MODE" \
  --device cuda \
  --camera-size 84 \
  2>&1 | tee "$LOG_PATH"
