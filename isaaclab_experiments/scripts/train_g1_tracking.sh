#!/usr/bin/env bash
# Train Unitree G1 whole-body motion tracking (Tracking-Flat-G1-v0).
#
# Usage:
#   bash scripts/train_g1_tracking.sh
#   MOTION=dance1_subject2 GPU=0 bash scripts/train_g1_tracking.sh
#   NUM_ENVS=4096 MAX_ITERS=20000 bash scripts/train_g1_tracking.sh
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
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-20000}"
SEED="${SEED:-42}"
MOTION="${MOTION:-walk1_subject1}"
MOTION_FILE="whole_body_tracking_reference_data/${MOTION}.npz"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/g1_tracking_launch"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/g1_track_${MOTION}_${STAMP}.log"

if [[ ! -f "$MOTION_FILE" ]]; then
  echo "ERROR: motion file not found: ${MOTION_FILE}" >&2
  echo "Available under whole_body_tracking_reference_data/: walk1_subject1, run1_subject2, ..." >&2
  exit 1
fi

echo "[$(date '+%F %T')] G1 tracking on GPU ${GPU}  motion=${MOTION}  num_envs=${NUM_ENVS}  iters=${MAX_ITERS}"
echo "  log -> ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU}" python isaaclab_fpo/scripts/train.py \
  --task Tracking-Flat-G1-v0 \
  --headless \
  --disable_fabric \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}" \
  --run_name "${STAMP}_g1_track_${MOTION}" \
  "env.commands.motion.motion_file=${MOTION_FILE}" \
  agent.device=cuda:0 \
  2>&1 | tee "${LOG_FILE}"
