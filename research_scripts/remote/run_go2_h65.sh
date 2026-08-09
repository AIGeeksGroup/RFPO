#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

STAGE="${STAGE:-construct}"
SOURCE_RUN="${SOURCE_RUN:-$PROJECT_ROOT/isaaclab_experiments/logs/isaaclab_fpo/unitree_go2_flat_flow/2026-08-08_16-24-22_go2_official_seed42_20260808}"
RESULT_DIR="${RESULT_DIR:-$RUNTIME_ROOT/results/go2_h65_tail_actor_average}"
LOG_DIR="${LOG_DIR:-$RUNTIME_ROOT/logs}"
CONTROL_CHECKPOINT="$SOURCE_RUN/model_1499.pt"
CANDIDATE_CHECKPOINT="$RESULT_DIR/model_h65_tail_actor_average.pt"
MANIFEST="$RESULT_DIR/construction_manifest.json"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

construct() {
  python isaaclab_fpo/scripts/create_tail_actor_average.py \
    --source-checkpoint "$SOURCE_RUN/model_1300.pt" \
    --source-checkpoint "$SOURCE_RUN/model_1350.pt" \
    --source-checkpoint "$SOURCE_RUN/model_1400.pt" \
    --source-checkpoint "$SOURCE_RUN/model_1450.pt" \
    --source-checkpoint "$SOURCE_RUN/model_1499.pt" \
    --output "$CANDIDATE_CHECKPOINT" --manifest "$MANIFEST" \
    2>&1 | tee "$LOG_DIR/go2_h65_construct.log"
}

evaluate() {
  local condition="$1"
  local mode="$2"
  local checkpoint output
  if [[ "$condition" == "control" ]]; then
    checkpoint="$CONTROL_CHECKPOINT"
  else
    checkpoint="$CANDIDATE_CHECKPOINT"
    python - "$MANIFEST" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    manifest = json.load(handle)
if not manifest.get("passed", False):
    raise SystemExit("H65 construction did not pass")
PY
  fi
  output="$RESULT_DIR/${condition}_${mode}.json"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite $output" >&2
    return 1
  fi
  python isaaclab_fpo/scripts/evaluate_checkpoint.py \
    --headless --device cuda:0 --checkpoint "$checkpoint" \
    --num-envs 256 --episodes 256 --seed 20261090 --source-seed 20261091 \
    --integration-method euler --sampling-steps 64 \
    --eval-modes "$mode" --output "$output" \
    2>&1 | tee "$LOG_DIR/go2_h65_${condition}_${mode}.log"
}

analyze() {
  python "$PROJECT_ROOT/experiments/go2-tail-swa/analyze.py" \
    --control-zero "$RESULT_DIR/control_zero.json" \
    --control-random "$RESULT_DIR/control_random.json" \
    --candidate-zero "$RESULT_DIR/candidate_zero.json" \
    --candidate-random "$RESULT_DIR/candidate_random.json" \
    --output "$RESULT_DIR/analysis.json" \
    2>&1 | tee "$LOG_DIR/go2_h65_analyze.log"
}

case "$STAGE" in
  construct) construct ;;
  control_zero) evaluate control zero ;;
  control_random) evaluate control random ;;
  candidate_zero) evaluate candidate zero ;;
  candidate_random) evaluate candidate random ;;
  analyze) analyze ;;
  all)
    construct
    evaluate control zero
    evaluate control random
    evaluate candidate zero
    evaluate candidate random
    analyze
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
