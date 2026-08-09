#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TRAIN_SEED="${TRAIN_SEED:-43}"
STAGE="${STAGE:-antithetic32}"
case "$TRAIN_SEED" in
  43)
    EVAL_SEED=20261243
    SOURCE_SEED=20261343
    SECONDARY_SOURCE_SEED=20261443
    ;;
  44)
    EVAL_SEED=20261244
    SOURCE_SEED=20261344
    SECONDARY_SOURCE_SEED=20261444
    ;;
  45)
    EVAL_SEED=20261245
    SOURCE_SEED=20261345
    SECONDARY_SOURCE_SEED=20261445
    ;;
  *)
    echo "TRAIN_SEED must be 43, 44, or 45" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

H69_ROOT="$RUNTIME_ROOT/results/go2_h69_antithetic_independent_seeds"
RESULT_ROOT="$RUNTIME_ROOT/results/go2_h73_antithetic_independent_equal_total_nfe"
RESULT_DIR="$RESULT_ROOT/seed_$TRAIN_SEED"
LOG_DIR="$RUNTIME_ROOT/logs"
MANIFEST="$H69_ROOT/training_seed${TRAIN_SEED}.json"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

checkpoint_from_manifest() {
  python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
checkpoint = Path(manifest["final_checkpoint"])
if not manifest.get("passed") or not checkpoint.is_file():
    raise SystemExit("invalid H69 training manifest")
print(checkpoint)
PY
}

evaluate() {
  local label="$1"
  local mode="$2"
  local checkpoint
  checkpoint="$(checkpoint_from_manifest)"
  python isaaclab_fpo/scripts/evaluate_checkpoint.py \
    --headless --device cuda:0 --checkpoint "$checkpoint" \
    --num-envs 512 --episodes 512 --seed "$EVAL_SEED" \
    --source-seed "$SOURCE_SEED" --secondary-source-seed "$SECONDARY_SOURCE_SEED" \
    --integration-method euler --sampling-steps 32 \
    --eval-modes "$mode" --output "$RESULT_DIR/${label}.json" \
    2>&1 | tee "$LOG_DIR/go2_h73_seed${TRAIN_SEED}_${label}.log"
}

case "$STAGE" in
  antithetic32) evaluate antithetic32 antithetic ;;
  zero32) evaluate zero32 zero ;;
  analyze)
    python "$PROJECT_ROOT/experiments/go2-antithetic-independent-equal-total-nfe/analyze.py" \
      --h69-dir "$H69_ROOT" --results-dir "$RESULT_ROOT" \
      --output "$RESULT_ROOT/analysis.json" \
      2>&1 | tee "$LOG_DIR/go2_h73_analyze.log"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
