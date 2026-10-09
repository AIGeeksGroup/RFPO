#!/usr/bin/env bash
# Train Boston Dynamics Spot flat velocity (online RL / FPO++ from scratch).
#
# Usage:
#   bash scripts/train_spot.sh
#   GPU=1 NUM_ENVS=16384 MAX_ITERS=1500 bash scripts/train_spot.sh
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
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERS="${MAX_ITERS:-1500}"   # paper / cfg default for Spot
SEED="${SEED:-42}"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/spot_launch"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/spot_${STAMP}.log"

echo "[$(date '+%F %T')] Spot on GPU ${GPU}  num_envs=${NUM_ENVS}  iters=${MAX_ITERS}"
echo "  log -> ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU}" python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Spot-v0 \
  --headless \
  --disable_fabric \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}" \
  --run_name "${STAMP}_spot" \
  agent.device=cuda:0 \
  2>&1 | tee "${LOG_FILE}"
