#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TRAIN_SEED="${TRAIN_SEED:?set TRAIN_SEED to 43, 44, or 45}"
case "$TRAIN_SEED" in
  43|44|45) ;;
  *)
    echo "TRAIN_SEED must be 43, 44, or 45" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

RUN_NAME="go2_h69_seed${TRAIN_SEED}"
LOG_ROOT="$PROJECT_ROOT/isaaclab_experiments/logs/isaaclab_fpo/unitree_go2_flat_flow"
MANIFEST_DIR="$RUNTIME_ROOT/results/go2_h69_antithetic_independent_seeds"
MANIFEST="$MANIFEST_DIR/training_seed${TRAIN_SEED}.json"
mkdir -p "$MANIFEST_DIR"

if [[ -e "$MANIFEST" ]]; then
  echo "Refusing to overwrite $MANIFEST" >&2
  exit 1
fi
if find "$LOG_ROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" | grep -q .; then
  echo "A run named $RUN_NAME already exists" >&2
  exit 1
fi

python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --max_iterations 1500 \
  --seed "$TRAIN_SEED" \
  --run_name "$RUN_NAME" \
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

python - "$TRAIN_SEED" "$RUN_DIR" "$CHECKPOINT" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

seed, run_dir, checkpoint, output = sys.argv[1:]
checkpoint_path = Path(checkpoint).resolve()
digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
result = {
    "hypothesis": "H69",
    "training_seed": int(seed),
    "num_envs": 4096,
    "iterations": 1500,
    "post_training_checkpoint_sweep": False,
    "run_dir": str(Path(run_dir).resolve()),
    "final_checkpoint": str(checkpoint_path),
    "final_checkpoint_sha256": digest,
    "final_checkpoint_bytes": checkpoint_path.stat().st_size,
    "passed": True,
}
with Path(output).open("x") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")
print(json.dumps(result, indent=2))
PY
