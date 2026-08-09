#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

STAGE="${STAGE:-zero64}"
SOURCE_RUN="${SOURCE_RUN:-$PROJECT_ROOT/isaaclab_experiments/logs/isaaclab_fpo/unitree_go2_flat_flow/2026-08-08_16-24-22_go2_official_seed42_20260808}"
CHECKPOINT="$SOURCE_RUN/model_1499.pt"
RESULT_DIR="$RUNTIME_ROOT/results/go2_h70_antithetic_equal_total_nfe"
LOG_DIR="$RUNTIME_ROOT/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

evaluate() {
  local label="$1"
  local mode="$2"
  local sampling_steps="$3"
  local output="$RESULT_DIR/${label}.json"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite $output" >&2
    return 1
  fi
  python isaaclab_fpo/scripts/evaluate_checkpoint.py \
    --headless --device cuda:0 --checkpoint "$CHECKPOINT" \
    --num-envs 256 --episodes 256 --seed 20261600 \
    --source-seed 20261601 --secondary-source-seed 20261603 \
    --integration-method euler --sampling-steps "$sampling_steps" \
    --eval-modes "$mode" --output "$output" \
    2>&1 | tee "$LOG_DIR/go2_h70_${label}.log"
}

case "$STAGE" in
  zero64) evaluate zero64 zero 64 ;;
  random64) evaluate random64 random 64 ;;
  antithetic32) evaluate antithetic32 antithetic 32 ;;
  antithetic64) evaluate antithetic64 antithetic 64 ;;
  analyze)
    python "$PROJECT_ROOT/experiments/go2-antithetic-equal-total-nfe/analyze.py" \
      --results-dir "$RESULT_DIR" \
      --output "$RESULT_DIR/analysis.json" \
      2>&1 | tee "$LOG_DIR/go2_h70_analyze.log"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
