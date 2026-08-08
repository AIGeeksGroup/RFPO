#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/isaaclab_experiments"
source source_env.sh

RUN_NAME="${RUN_NAME:-go2_official_seed42_$(timestamp)}"
SEED="${SEED:-42}"

python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --max_iterations 1500 \
  --seed "$SEED" \
  --run_name "$RUN_NAME"

