#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

STAGE="${STAGE:-zero32}"
SOURCE_RUN="${SOURCE_RUN:-$PROJECT_ROOT/isaaclab_experiments/logs/isaaclab_fpo/unitree_go2_flat_flow/2026-08-08_16-24-22_go2_official_seed42_20260808}"
CHECKPOINT="$SOURCE_RUN/model_1499.pt"
H70_DIR="$RUNTIME_ROOT/results/go2_h70_antithetic_equal_total_nfe"
RESULT_DIR="$RUNTIME_ROOT/results/go2_h72_antithetic_reduced_zero_control"
LOG_DIR="$RUNTIME_ROOT/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

case "$STAGE" in
  zero32)
    python isaaclab_fpo/scripts/evaluate_checkpoint.py \
      --headless --device cuda:0 --checkpoint "$CHECKPOINT" \
      --num-envs 256 --episodes 256 --seed 20261600 \
      --source-seed 20261601 --secondary-source-seed 20261603 \
      --integration-method euler --sampling-steps 32 \
      --eval-modes zero --output "$RESULT_DIR/zero32.json" \
      2>&1 | tee "$LOG_DIR/go2_h72_zero32.log"
    ;;
  analyze)
    python "$PROJECT_ROOT/experiments/go2-antithetic-reduced-zero-control/analyze.py" \
      --h70-dir "$H70_DIR" --zero32 "$RESULT_DIR/zero32.json" \
      --output "$RESULT_DIR/analysis.json" \
      2>&1 | tee "$LOG_DIR/go2_h72_analyze.log"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
