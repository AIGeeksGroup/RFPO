#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${FPO_PROJECT_ROOT:-$HOME/workspace/code/fpo-control}"
RUNTIME_ROOT="${FPO_RUNTIME_ROOT:-$HOME/workspace/outputs/fpo-control}"
FPO_GPU="${FPO_GPU:-1}"

export CUDA_VISIBLE_DEVICES="$FPO_GPU"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/workspace/scratch/fpo-cache}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$RUNTIME_ROOT/logs" "$RUNTIME_ROOT/results" "$XDG_CACHE_HOME"

timestamp() {
  date +%Y%m%d_%H%M%S
}

