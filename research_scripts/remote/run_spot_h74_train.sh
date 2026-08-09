#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

RUN_NAME="spot_h74_seed42"
LOG_ROOT="$PROJECT_ROOT/isaaclab_experiments/logs/isaaclab_fpo/spot_flat_flow"
RESULT_DIR="$RUNTIME_ROOT/results/spot_h74_antithetic_cross_task"
MANIFEST="$RESULT_DIR/training_seed42.json"
mkdir -p "$RESULT_DIR"

if [[ -e "$MANIFEST" ]]; then
  echo "Refusing to overwrite $MANIFEST" >&2
  exit 1
fi
if find "$LOG_ROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" 2>/dev/null | grep -q .; then
  echo "A run named $RUN_NAME already exists" >&2
  exit 1
fi

python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Spot-v0 \
  --headless --device cuda:0 --num_envs 4096 \
  --max_iterations 1500 --seed 42 --run_name "$RUN_NAME" \
  agent.enable_post_training_eval=False

mapfile -t run_dirs < <(
  find "$LOG_ROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" | sort
)
if [[ "${#run_dirs[@]}" -ne 1 ]]; then
  echo "Expected exactly one completed run for $RUN_NAME" >&2
  exit 1
fi
RUN_DIR="${run_dirs[0]}"
CHECKPOINT="$RUN_DIR/model_1499.pt"
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Missing final checkpoint $CHECKPOINT" >&2
  exit 1
fi

python - "$RUN_DIR" "$CHECKPOINT" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_dir, checkpoint, output = sys.argv[1:]
checkpoint_path = Path(checkpoint).resolve()
result = {
    "hypothesis": "H74",
    "training_seed": 42,
    "task": "Isaac-Velocity-Flat-Spot-v0",
    "num_envs": 4096,
    "iterations": 1500,
    "post_training_checkpoint_sweep": False,
    "run_dir": str(Path(run_dir).resolve()),
    "final_checkpoint": str(checkpoint_path),
    "final_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    "final_checkpoint_bytes": checkpoint_path.stat().st_size,
    "passed": True,
}
with Path(output).open("x") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")
print(json.dumps(result, indent=2))
PY
