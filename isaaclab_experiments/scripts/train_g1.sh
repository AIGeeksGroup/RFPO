#!/usr/bin/env bash
# Train Unitree G1 flat velocity (online RL / FPO++ from scratch).
#
# Usage:
#   bash scripts/train_g1.sh
#   GPU=0 NUM_ENVS=8192 MAX_ITERS=2000 bash scripts/train_g1.sh
#   # optional reflow variant (see G1_FPO_VARIANTS):
#   FPO_VARIANT=reward_aware GPU=1 bash scripts/train_g1.sh
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
# G1 is heavier than Go2/Spot; 8192 is a safer default on 80GB (same as H1).
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-2000}"   # paper / cfg default for G1
SEED="${SEED:-42}"
FPO_VARIANT="${FPO_VARIANT:-}"   # empty = default baseline cfg
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/g1_launch"
mkdir -p "$LOG_DIR"
SUFFIX="${FPO_VARIANT:-baseline}"
LOG_FILE="${LOG_DIR}/g1_${SUFFIX}_${STAMP}.log"

EXTRA_ARGS=()
if [[ -n "$FPO_VARIANT" ]]; then
  EXTRA_ARGS+=(--fpo_variant "$FPO_VARIANT")
fi

echo "[$(date '+%F %T')] G1 on GPU ${GPU}  num_envs=${NUM_ENVS}  iters=${MAX_ITERS}  variant=${SUFFIX}"
echo "  log -> ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU}" python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-G1-v0 \
  --headless \
  --disable_fabric \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}" \
  --run_name "${STAMP}_g1_${SUFFIX}" \
  "${EXTRA_ARGS[@]}" \
  agent.device=cuda:0 \
  2>&1 | tee "${LOG_FILE}"
