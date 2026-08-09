#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

SOURCE_RUN="${SOURCE_RUN:-$PROJECT_ROOT/isaaclab_experiments/logs/isaaclab_fpo/unitree_go2_flat_flow/2026-08-08_16-24-22_go2_official_seed42_20260808}"
CHECKPOINT="$SOURCE_RUN/model_1499.pt"
RESULT_DIR="$RUNTIME_ROOT/results/go2_h71_antithetic_batched_throughput"
LOG_DIR="$RUNTIME_ROOT/logs"
OUTPUT="$RESULT_DIR/benchmark.json"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite $OUTPUT" >&2
  exit 1
fi

python isaaclab_fpo/scripts/benchmark_antithetic_throughput.py \
  --headless --device cuda:0 --checkpoint "$CHECKPOINT" \
  --num-envs 4096 --seed 20261610 \
  --warmup-calls 10 --blocks 5 --calls-per-block 20 \
  --output "$OUTPUT" \
  2>&1 | tee "$LOG_DIR/go2_h71_antithetic_batched_throughput.log"
