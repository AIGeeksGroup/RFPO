#!/usr/bin/env bash
# Smoke-test all four reflow idea variants in Isaac (2 iterations each).
set -eo pipefail

ROOT="/workspace/fpo-control/isaaclab_experiments"
cd "$ROOT"
source source_env.sh

TASK="Isaac-Velocity-Flat-Unitree-Go2-v0"
COMMON=(
  python isaaclab_fpo/scripts/train.py
  --task "$TASK"
  --headless
  --disable_fabric
  --num_envs 64
  --max_iterations 2
)

run_variant() {
  local gpu="$1"
  local variant="$2"
  local run_name="smoke_${variant}"
  echo "[GPU${gpu}] smoke test: ${variant}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${COMMON[@]}" \
    --fpo_variant "${variant}" \
    --run_name "${run_name}" \
    agent.device=cuda:0
}

run_variant 0 reward_aware &
pid0=$!
run_variant 1 adaptive_compute &
pid1=$!
wait "$pid0" "$pid1"

run_variant 0 fpo_operator &
pid0=$!
run_variant 1 theory &
pid1=$!
wait "$pid0" "$pid1"

echo "Isaac smoke tests finished."
