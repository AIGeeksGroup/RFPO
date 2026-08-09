#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

STAGE="${STAGE:-zero64}"
cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

RESULT_DIR="$RUNTIME_ROOT/results/spot_h74_antithetic_cross_task"
MANIFEST="$RESULT_DIR/training_seed42.json"
LOG_DIR="$RUNTIME_ROOT/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

checkpoint_from_manifest() {
  python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
checkpoint = Path(manifest["final_checkpoint"])
if not manifest.get("passed") or not checkpoint.is_file():
    raise SystemExit("invalid H74 training manifest")
print(checkpoint)
PY
}

evaluate() {
  local label="$1"
  local mode="$2"
  local steps="$3"
  local checkpoint
  checkpoint="$(checkpoint_from_manifest)"
  python isaaclab_fpo/scripts/evaluate_checkpoint.py \
    --headless --device cuda:0 --checkpoint "$checkpoint" \
    --task Isaac-Velocity-Flat-Spot-v0 \
    --num-envs 256 --episodes 256 --seed 20261640 \
    --source-seed 20261641 --secondary-source-seed 20261642 \
    --integration-method euler --sampling-steps "$steps" \
    --eval-modes "$mode" --output "$RESULT_DIR/${label}.json" \
    2>&1 | tee "$LOG_DIR/spot_h74_${label}.log"
}

case "$STAGE" in
  zero64) evaluate zero64 zero 64 ;;
  zero32) evaluate zero32 zero 32 ;;
  random64) evaluate random64 random 64 ;;
  iid_pair32) evaluate iid_pair32 iid_pair 32 ;;
  antithetic32) evaluate antithetic32 antithetic 32 ;;
  analyze)
    python "$PROJECT_ROOT/experiments/spot-antithetic-cross-task/analyze.py" \
      --results-dir "$RESULT_DIR" --output "$RESULT_DIR/analysis.json" \
      2>&1 | tee "$LOG_DIR/spot_h74_analyze.log"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
