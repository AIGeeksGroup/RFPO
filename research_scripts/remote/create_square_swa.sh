#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

SOURCE_ROOT="${SOURCE_ROOT:-$RUNTIME_ROOT/results/square_h43_control_osmesa_seed20260916/checkpoints}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-$RUNTIME_ROOT/results/square_h44_swa_from_h43_control}"
if [[ -e "$OUTPUT_CHECKPOINT" ]]; then
  echo "Refusing to overwrite existing H44 checkpoint: $OUTPUT_CHECKPOINT" >&2
  exit 1
fi

for step in 10240 15360 20480 25600; do
  if [[ ! -f "$SOURCE_ROOT/step_${step}/policy/model.safetensors" ]]; then
    echo "Missing H44 source checkpoint: $SOURCE_ROOT/step_${step}" >&2
    exit 1
  fi
done

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

python average_policy_checkpoints.py \
  --source-checkpoint "$SOURCE_ROOT/step_10240" \
  --source-checkpoint "$SOURCE_ROOT/step_15360" \
  --source-checkpoint "$SOURCE_ROOT/step_20480" \
  --source-checkpoint "$SOURCE_ROOT/step_25600" \
  --output-checkpoint "$OUTPUT_CHECKPOINT"
