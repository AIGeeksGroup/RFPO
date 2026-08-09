#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

STAGE="${STAGE:-control}"
SOURCE_RUN="${SOURCE_RUN:-$PROJECT_ROOT/isaaclab_experiments/logs/isaaclab_fpo/unitree_go2_flat_flow/2026-08-08_16-24-22_go2_official_seed42_20260808}"
CHECKPOINT="$SOURCE_RUN/model_1499.pt"
SCREEN_DIR="$RUNTIME_ROOT/results/go2_h66_antithetic_source_ensemble"
RESULT_DIR="${RESULT_DIR:-$SCREEN_DIR/confirmation}"
LOG_DIR="${LOG_DIR:-$RUNTIME_ROOT/logs}"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

python - "$SCREEN_DIR/geometry.json" "$SCREEN_DIR/analysis.json" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    with open(path) as handle:
        result = json.load(handle)
    if not result.get("passed", False):
        raise SystemExit(f"H66 prerequisite did not pass: {path}")
PY

evaluate() {
  local mode="$1"
  local output="$RESULT_DIR/${mode}.json"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite $output" >&2
    return 1
  fi
  python isaaclab_fpo/scripts/evaluate_checkpoint.py \
    --headless --device cuda:0 --checkpoint "$CHECKPOINT" \
    --num-envs 4096 --episodes 4096 --seed 20261103 --source-seed 20261104 \
    --integration-method euler --sampling-steps 64 \
    --eval-modes "$mode" --output "$output" \
    2>&1 | tee "$LOG_DIR/go2_h66_confirmation_${mode}.log"
}

case "$STAGE" in
  control) evaluate random ;;
  candidate) evaluate antithetic ;;
  analyze)
    python "$PROJECT_ROOT/experiments/go2-antithetic-source-ensemble/analyze_confirmation.py" \
      --control "$RESULT_DIR/random.json" \
      --candidate "$RESULT_DIR/antithetic.json" \
      --output "$RESULT_DIR/analysis.json" \
      2>&1 | tee "$LOG_DIR/go2_h66_confirmation_analyze.log"
    ;;
  *)
    echo "Unknown STAGE=$STAGE (expected control, candidate, or analyze)" >&2
    exit 2
    ;;
esac
