#!/usr/bin/env bash
# Launch Go2 plain reflow + all_ideas (reward_aware + adaptive_compute + theory).
#
# Usage:
#   GPU_REFLOW=0 GPU_ALL_IDEAS=2 bash scripts/launch_go2_reflow_missing.sh
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-/nix/plsy/tmp/tmpdir}"
mkdir -p "$TMPDIR"

TASK="Isaac-Velocity-Flat-Unitree-Go2-v0"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/reflow_missing_2500"
mkdir -p "$LOG_DIR"
echo "$STAMP" > "${LOG_DIR}/stamp.txt"
OUT="${LOG_DIR}/launch.out"

GPU_REFLOW="${GPU_REFLOW:-0}"
GPU_ALL_IDEAS="${GPU_ALL_IDEAS:-2}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERS="${MAX_ITERS:-2500}"

launch() {
  local gpu="$1"
  local variant="$2"
  local log_file="${LOG_DIR}/${variant}.log"
  echo "[$(date '+%F %T')] Launching Go2 ${variant} on GPU ${gpu} (num_envs=${NUM_ENVS}, max_iters=${MAX_ITERS}) -> ${log_file}" | tee -a "$OUT"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python isaaclab_fpo/scripts/train.py \
    --task "$TASK" \
    --headless \
    --disable_fabric \
    --num_envs "$NUM_ENVS" \
    --fpo_variant "$variant" \
    --max_iterations "$MAX_ITERS" \
    --run_name "${STAMP}_${variant}" \
    agent.device=cuda:0 \
    agent.enable_post_training_eval=false \
    > "${log_file}" 2>&1 &
  local pid=$!
  echo "${pid} ${gpu} ${variant}" >> "${LOG_DIR}/pids.txt"
  echo "  PID=${pid}" | tee -a "$OUT"
}

: > "${LOG_DIR}/pids.txt"
launch "$GPU_REFLOW" reflow
launch "$GPU_ALL_IDEAS" all_ideas

echo "[$(date '+%F %T')] Both jobs launched. stamp=${STAMP} logs=${LOG_DIR}" | tee -a "$OUT"
