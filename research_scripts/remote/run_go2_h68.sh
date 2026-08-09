#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

STAGE="${STAGE:-zero}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-500}"
SOURCE_RUN="${SOURCE_RUN:-$PROJECT_ROOT/isaaclab_experiments/logs/isaaclab_fpo/unitree_go2_flat_flow/2026-08-08_16-24-22_go2_official_seed42_20260808}"
RESULT_ROOT="${RESULT_ROOT:-$RUNTIME_ROOT/results/go2_h68_antithetic_cross_checkpoint}"
LOG_DIR="${LOG_DIR:-$RUNTIME_ROOT/logs}"
mkdir -p "$RESULT_ROOT" "$LOG_DIR"

case "$CHECKPOINT_ITER" in
  500)
    EVAL_SEED=20261120
    SOURCE_SEED=20261130
    SECONDARY_SOURCE_SEED=20261140
    ;;
  1000)
    EVAL_SEED=20261121
    SOURCE_SEED=20261131
    SECONDARY_SOURCE_SEED=20261141
    ;;
  *)
    echo "CHECKPOINT_ITER must be 500 or 1000" >&2
    exit 2
    ;;
esac

evaluate() {
  local mode="$1"
  local result_dir="$RESULT_ROOT/checkpoint_$CHECKPOINT_ITER"
  local output="$result_dir/${mode}.json"
  mkdir -p "$result_dir"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite $output" >&2
    return 1
  fi
  python isaaclab_fpo/scripts/evaluate_checkpoint.py \
    --headless --device cuda:0 --checkpoint "$SOURCE_RUN/model_$CHECKPOINT_ITER.pt" \
    --num-envs 128 --episodes 128 --seed "$EVAL_SEED" \
    --source-seed "$SOURCE_SEED" --secondary-source-seed "$SECONDARY_SOURCE_SEED" \
    --integration-method euler --sampling-steps 64 \
    --eval-modes "$mode" --output "$output" \
    2>&1 | tee "$LOG_DIR/go2_h68_${CHECKPOINT_ITER}_${mode}.log"
}

case "$STAGE" in
  zero|iid_pair|antithetic) evaluate "$STAGE" ;;
  analyze)
    python "$PROJECT_ROOT/experiments/go2-antithetic-cross-checkpoint/analyze.py" \
      --results-dir "$RESULT_ROOT" \
      --h67-analysis "$RUNTIME_ROOT/results/go2_h67_antithetic_attribution/analysis.json" \
      --output "$RESULT_ROOT/analysis.json" \
      2>&1 | tee "$LOG_DIR/go2_h68_analyze.log"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
