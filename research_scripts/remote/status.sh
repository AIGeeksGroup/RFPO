#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

find "$PROJECT_ROOT/isaaclab_experiments/logs" -maxdepth 5 -type f \
  \( -name '*.out' -o -name 'events.out.tfevents.*' -o -name 'model_*.pt' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -30

