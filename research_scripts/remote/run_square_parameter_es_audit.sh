#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
SQUARE_CHECKPOINT="${SQUARE_CHECKPOINT:-$CHECKPOINT_ROOT/trc7rbt0_step_110000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/workspace/outputs/fpo-control}"
MODE="${MODE:-smoke}"
FPO_GPU="${FPO_GPU:-1}"
FPO_RENDER_GPU="${FPO_RENDER_GPU:-$FPO_GPU}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/manipulation_experiments/thirdparty/miniconda3/envs/fpo_manipulation/bin/python}"
OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"

case "$MODE" in
  smoke|audit) ;;
  *) echo "MODE must be smoke or audit" >&2; exit 1 ;;
esac
if [[ ! -f "$SQUARE_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Square checkpoint: $SQUARE_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing FPO Python environment: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$OSMESA_LIB/libOSMesa.so.8" ]]; then
  echo "Missing OSMesa runtime: $OSMESA_LIB/libOSMesa.so.8" >&2
  exit 1
fi

RUN_DIR="$OUTPUT_ROOT/h53_paired_parameter_es_square"
mkdir -p "$RUN_DIR"

cd "$PROJECT_ROOT/manipulation_experiments"
export CUDA_VISIBLE_DEVICES="$FPO_GPU"
export FPO_RENDER_GPU
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" audit_paired_parameter_es.py \
  --local-checkpoint-path "$SQUARE_CHECKPOINT" \
  --output-dir "$RUN_DIR" \
  --mode "$MODE" \
  --device cuda \
  2>&1 | tee -a "$RUN_DIR/${MODE}.log"
