#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VIVIDEX_HEADLESS_NO_RENDER=1
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-0}"
export SEQ="${SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"

mkdir -p norm_trajectories/rm75_inspire_right

echo "[rm75-smoke] seq=${SEQ}"
nvidia-smi || true

python tools/retarget_rm75_inspire_reference.py \
  --src "norm_trajectories/${SEQ}.npz" \
  --dst "norm_trajectories/rm75_inspire_right/${SEQ}.npz" \
  --overwrite

python tools/diagnose_rm75_reference_alignment.py \
  --seq-name "${SEQ}" \
  --robot-name rm75_inspire_right \
  --norm-traj

python tools/scripted_rm75_grasp_smoke.py \
  --seq-name "${SEQ}" \
  --robot-name rm75_inspire_right \
  --norm-traj \
  --mode track_reference \
  --close-val "${RM75_CLOSE_VAL:-0.9}" \
  --steps "${RM75_SMOKE_STEPS:-80}"
