#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

PHYSICAL_GPU="${PHYSICAL_GPU:-1}"
STAGE="${STAGE:-geometry}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/isaaclab_experiments/logs/isaaclab_fpo/unitree_go2_flat_flow/2026-08-08_16-24-22_go2_official_seed42_20260808/model_1499.pt}"
RESULT_DIR="${RESULT_DIR:-$RUNTIME_ROOT/results/go2_h60_equal_nfe_midpoint}"
LOG_DIR="${LOG_DIR:-$RUNTIME_ROOT/logs}"
mkdir -p "$RESULT_DIR" "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"

case "$STAGE" in
  geometry)
    python isaaclab_fpo/scripts/audit_flow_integration.py \
      --headless --device cuda:0 --checkpoint "$CHECKPOINT" \
      --num-envs 256 --seed 20261060 --rollout-steps 8 \
      --output "$RESULT_DIR/geometry.json" \
      2>&1 | tee "$LOG_DIR/go2_h60_geometry.log"
    ;;
  control|candidate)
    python - "$RESULT_DIR/geometry.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    geometry = json.load(handle)
if not geometry.get("passed", False):
    raise SystemExit("H60 Stage A did not pass; refusing to start reward evaluation")
PY
    if [[ "$STAGE" == "control" ]]; then
      METHOD="euler"
      STEPS=64
    else
      METHOD="midpoint"
      STEPS=32
    fi
    python isaaclab_fpo/scripts/evaluate_checkpoint.py \
      --headless --device cuda:0 --checkpoint "$CHECKPOINT" \
      --num-envs 50 --episodes 50 --seed 20261061 --source-seed 20261060 \
      --integration-method "$METHOD" --sampling-steps "$STEPS" \
      --eval-modes zero random --output "$RESULT_DIR/$STAGE.json" \
      2>&1 | tee "$LOG_DIR/go2_h60_$STAGE.log"
    ;;
  analyze)
    python "$PROJECT_ROOT/experiments/go2-equal-nfe-midpoint/analyze.py" \
      --control "$RESULT_DIR/control.json" --candidate "$RESULT_DIR/candidate.json" \
      --output "$RESULT_DIR/analysis.json"
    ;;
  *)
    echo "Unknown STAGE=$STAGE (expected geometry, control, candidate, or analyze)" >&2
    exit 2
    ;;
esac
