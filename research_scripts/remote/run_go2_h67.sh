#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

STAGE="${STAGE:-geometry}"
SOURCE_RUN="${SOURCE_RUN:-$PROJECT_ROOT/isaaclab_experiments/logs/isaaclab_fpo/unitree_go2_flat_flow/2026-08-08_16-24-22_go2_official_seed42_20260808}"
CHECKPOINT="$SOURCE_RUN/model_1499.pt"
RESULT_DIR="${RESULT_DIR:-$RUNTIME_ROOT/results/go2_h67_antithetic_attribution}"
LOG_DIR="${LOG_DIR:-$RUNTIME_ROOT/logs}"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

require_geometry() {
  python - "$RESULT_DIR/geometry.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    result = json.load(handle)
if not result.get("passed", False):
    raise SystemExit("H67 mechanism gate did not pass")
PY
}

evaluate() {
  local mode="$1"
  local output="$RESULT_DIR/${mode}.json"
  require_geometry
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite $output" >&2
    return 1
  fi
  python isaaclab_fpo/scripts/evaluate_checkpoint.py \
    --headless --device cuda:0 --checkpoint "$CHECKPOINT" \
    --num-envs 256 --episodes 256 --seed 20261110 \
    --source-seed 20261111 --secondary-source-seed 20261112 \
    --integration-method euler --sampling-steps 64 \
    --eval-modes "$mode" --output "$output" \
    2>&1 | tee "$LOG_DIR/go2_h67_${mode}.log"
}

case "$STAGE" in
  geometry)
    python isaaclab_fpo/scripts/audit_source_pair_attribution.py \
      --headless --device cuda:0 --checkpoint "$CHECKPOINT" \
      --num-envs 256 --rollout-steps 8 --seed 20261110 \
      --source-seed 20261111 --secondary-source-seed 20261112 \
      --output "$RESULT_DIR/geometry.json" \
      2>&1 | tee "$LOG_DIR/go2_h67_geometry.log"
    ;;
  zero|random|iid_pair|antithetic) evaluate "$STAGE" ;;
  analyze)
    python "$PROJECT_ROOT/experiments/go2-antithetic-attribution/analyze.py" \
      --zero "$RESULT_DIR/zero.json" \
      --random "$RESULT_DIR/random.json" \
      --iid-pair "$RESULT_DIR/iid_pair.json" \
      --antithetic "$RESULT_DIR/antithetic.json" \
      --output "$RESULT_DIR/analysis.json" \
      2>&1 | tee "$LOG_DIR/go2_h67_analyze.log"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
