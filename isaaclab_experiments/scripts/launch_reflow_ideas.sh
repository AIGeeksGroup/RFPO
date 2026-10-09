#!/usr/bin/env bash
# Launch four reflow idea experiments on two GPUs (1500 iterations each).
set -eo pipefail

ROOT="/workspace/fpo-control/isaaclab_experiments"
cd "$ROOT"
source source_env.sh

TASK="Isaac-Velocity-Flat-Unitree-Go2-v0"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/reflow_ideas_launch"
mkdir -p "$LOG_DIR"

COMMON=(
  python isaaclab_fpo/scripts/train.py
  --task "$TASK"
  --headless
  --disable_fabric
  --max_iterations 1500
)

launch() {
  local gpu="$1"
  local variant="$2"
  local run_name="${STAMP}_${variant}"
  local log_file="${LOG_DIR}/${variant}.log"
  echo "Launching ${variant} on GPU ${gpu} -> ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${COMMON[@]}" \
    --fpo_variant "${variant}" \
    --run_name "${run_name}" \
    agent.device=cuda:0 \
    > "${log_file}" 2>&1 &
  echo "$!" >> "${LOG_DIR}/pids.txt"
}

: > "${LOG_DIR}/pids.txt"

launch 0 reward_aware
launch 1 adaptive_compute
wait

launch 0 fpo_operator
launch 1 theory
wait

echo "All four jobs finished. Logs in ${LOG_DIR}/"
