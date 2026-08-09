#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

STAGE="${STAGE:-seed42}"
RESULT_DIR="$RUNTIME_ROOT/results/go2_h75_antithetic_affine_mechanism"
LOG_DIR="$RUNTIME_ROOT/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

checkpoint_for_seed() {
  local training_seed="$1"
  python - "$training_seed" "$RUNTIME_ROOT" <<'PY'
import json
import sys
from pathlib import Path

seed = int(sys.argv[1])
root = Path(sys.argv[2])
if seed == 42:
    artifact = root / "results/go2_h67_antithetic_attribution/geometry.json"
    checkpoint = json.loads(artifact.read_text())["checkpoint"]
else:
    artifact = root / f"results/go2_h69_antithetic_independent_seeds/training_seed{seed}.json"
    checkpoint = json.loads(artifact.read_text())["final_checkpoint"]
path = Path(checkpoint)
if not path.is_file():
    raise SystemExit(f"missing checkpoint: {path}")
print(path)
PY
}

audit_seed() {
  local training_seed="$1"
  local checkpoint
  checkpoint="$(checkpoint_for_seed "$training_seed")"
  cd "$PROJECT_ROOT/isaaclab_experiments"
  source source_env.sh
  python isaaclab_fpo/scripts/audit_affine_antisymmetry.py \
    --headless --device cuda:0 --checkpoint "$checkpoint" \
    --training-seed "$training_seed" \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num-envs 128 --seed 20261650 \
    --source-seed 20261651 --secondary-source-seed 20261652 \
    --rollout-steps 8 --sampling-steps 32 \
    --output "$RESULT_DIR/seed${training_seed}.json" \
    2>&1 | tee "$LOG_DIR/go2_h75_seed${training_seed}.log"
}

case "$STAGE" in
  seed42) audit_seed 42 ;;
  seed43) audit_seed 43 ;;
  seed44) audit_seed 44 ;;
  seed45) audit_seed 45 ;;
  analyze)
    python "$PROJECT_ROOT/experiments/go2-antithetic-affine-mechanism/analyze.py" \
      --results-dir "$RESULT_DIR" --output "$RESULT_DIR/analysis.json" \
      2>&1 | tee "$LOG_DIR/go2_h75_analyze.log"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
