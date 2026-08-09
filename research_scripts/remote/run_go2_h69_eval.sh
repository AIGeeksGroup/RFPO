#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TRAIN_SEED="${TRAIN_SEED:-43}"
STAGE="${STAGE:-zero}"
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

RESULT_ROOT="$RUNTIME_ROOT/results/go2_h69_antithetic_independent_seeds"
RESULT_DIR="$RESULT_ROOT/seed_$TRAIN_SEED"
MANIFEST="$RESULT_ROOT/training_seed${TRAIN_SEED}.json"
LOG_DIR="$RUNTIME_ROOT/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

checkpoint_from_manifest() {
  python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1]) as handle:
    manifest = json.load(handle)
if not manifest.get("passed", False):
    raise SystemExit("training manifest did not pass")
checkpoint = Path(manifest["final_checkpoint"])
if not checkpoint.is_file():
    raise SystemExit(f"missing checkpoint: {checkpoint}")
print(checkpoint)
PY
}

evaluate() {
  local mode="$1"
  local checkpoint
  local output="$RESULT_DIR/${mode}.json"
  checkpoint="$(checkpoint_from_manifest)"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite $output" >&2
    return 1
  fi
  python isaaclab_fpo/scripts/evaluate_checkpoint.py \
    --headless --device cuda:0 --checkpoint "$checkpoint" \
    --num-envs 512 --episodes 512 --seed "$EVAL_SEED" \
    --source-seed "$SOURCE_SEED" --secondary-source-seed "$SECONDARY_SOURCE_SEED" \
    --integration-method euler --sampling-steps 64 \
    --eval-modes "$mode" --output "$output" \
    2>&1 | tee "$LOG_DIR/go2_h69_seed${TRAIN_SEED}_${mode}.log"
}

case "$STAGE" in
  zero|random|iid_pair|antithetic) evaluate "$STAGE" ;;
  analyze)
    python "$PROJECT_ROOT/experiments/go2-antithetic-independent-seeds/analyze.py" \
      --results-dir "$RESULT_ROOT" \
      --h67-analysis "$RUNTIME_ROOT/results/go2_h67_antithetic_attribution/analysis.json" \
      --output "$RESULT_ROOT/analysis.json" \
      2>&1 | tee "$LOG_DIR/go2_h69_analyze.log"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
