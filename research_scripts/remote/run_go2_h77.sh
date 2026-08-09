#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

STAGE="${STAGE:-all}"
RESULT_DIR="$RUNTIME_ROOT/results/go2_h77_mirrored_rollout_variance"
LOG_DIR="$RUNTIME_ROOT/logs"
EVALUATION="$RESULT_DIR/evaluation.json"
ANALYSIS="$RESULT_DIR/analysis.json"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

checkpoint_path() {
  python - "$RUNTIME_ROOT" <<'PY'
import json
import sys
from pathlib import Path

artifact = Path(sys.argv[1]) / "results/go2_h67_antithetic_attribution/geometry.json"
checkpoint = Path(json.loads(artifact.read_text())["checkpoint"])
if not checkpoint.is_file():
    raise SystemExit(f"missing checkpoint: {checkpoint}")
print(checkpoint)
PY
}

evaluate() {
  if [[ -e "$EVALUATION" ]]; then
    echo "Refusing to overwrite $EVALUATION" >&2
    exit 1
  fi
  local checkpoint
  checkpoint="$(checkpoint_path)"
  cd "$PROJECT_ROOT/isaaclab_experiments"
  source source_env.sh
  python isaaclab_fpo/scripts/evaluate_checkpoint.py \
    --headless --device cuda:0 --checkpoint "$checkpoint" \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num-envs 256 --episodes 256 --seed 20261650 \
    --source-seed 20261651 --secondary-source-seed 20261652 \
    --integration-method euler --sampling-steps 32 \
    --eval-modes zero random negative_random secondary_random \
    --output "$EVALUATION" \
    2>&1 | tee "$LOG_DIR/go2_h77_evaluate.log"
}

analyze() {
  if [[ -e "$ANALYSIS" ]]; then
    echo "Refusing to overwrite $ANALYSIS" >&2
    exit 1
  fi
  python "$PROJECT_ROOT/experiments/go2-mirrored-rollout-variance/analyze.py" \
    --input "$EVALUATION" --output "$ANALYSIS" \
    2>&1 | tee "$LOG_DIR/go2_h77_analyze.log"
}

case "$STAGE" in
  evaluate) evaluate ;;
  analyze) analyze ;;
  all) evaluate; analyze ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
