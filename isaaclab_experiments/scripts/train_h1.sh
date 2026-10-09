#!/usr/bin/env bash
# Train Unitree H1 flat velocity (online RL / FPO++ from scratch).
#
# Usage:
#   bash scripts/train_h1.sh
#   GPU=3 NUM_ENVS=8192 MAX_ITERS=2000 bash scripts/train_h1.sh
#
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-/nix/plsy/tmp/tmpdir}"
mkdir -p "$TMPDIR"

GPU="${GPU:-0}"
# H1 is heavier than Go2/Spot; 8192 is a safer default on 80GB; raise if VRAM allows.
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-2000}"   # paper / cfg default for H1
SEED="${SEED:-42}"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/h1_launch"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/h1_${STAMP}.log"

echo "[$(date '+%F %T')] H1 on GPU ${GPU}  num_envs=${NUM_ENVS}  iters=${MAX_ITERS}"
echo "  log -> ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU}" python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-H1-v0 \
  --headless \
  --disable_fabric \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}" \
  --run_name "${STAMP}_h1" \
  agent.device=cuda:0 \
  2>&1 | tee "${LOG_FILE}"
