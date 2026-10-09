#!/usr/bin/env bash
# Resume the two best idea runs for +500 iterations.
set -e

ROOT="/workspace/fpo-control/isaaclab_experiments"
cd "$ROOT"
# shellcheck disable=SC1091
source source_env.sh

LOG_DIR="${ROOT}/logs/isaaclab_fpo/reflow_ideas_launch"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"

launch() {
  local gpu="$1"
  local variant="$2"
  local load_run="$3"
  local run_name="${STAMP}_${variant}_v2"
  local log_file="${LOG_DIR}/${variant}_v2.log"

  echo "Launching ${variant}_v2 on GPU ${gpu} (resume +500)"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --headless \
    --disable_fabric \
    --resume \
    --load_run "${load_run}" \
    --checkpoint model_1499.pt \
    --fpo_variant "${variant}" \
    --run_name "${run_name}" \
    --max_iterations 500 \
    agent.device=cuda:0 \
    > "${log_file}" 2>&1 &
  echo $! > "${LOG_DIR}/${variant}_v2.pid"
  echo "  pid=$(cat "${LOG_DIR}/${variant}_v2.pid")  log=${log_file}"
}

launch 0 adaptive_compute "2026-08-20_00-14-03_2026-08-20_00-14-00_adaptive_compute"
sleep 20
launch 1 reward_aware "2026-08-20_00-14-03_2026-08-20_00-14-00_reward_aware"

echo "Both resume jobs launched."
