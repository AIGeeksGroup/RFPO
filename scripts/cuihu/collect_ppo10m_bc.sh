#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

NUM_TRAJS="${NUM_TRAJS:-100}"

nvidia-smi || true
python tools/collect_fpo_bc_dataset.py \
  --checkpoint_file "${PPO_10M}" \
  --config_dir "${PPO_RUN}" \
  -o "${BC_DATASET}" \
  -n "${NUM_TRAJS}" \
  --stage 2 \
  --success_metric sr10 \
  --norm_traj

