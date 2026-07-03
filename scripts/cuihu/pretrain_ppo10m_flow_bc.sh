#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

nvidia-smi || true
python tools/pretrain_fpo_bc.py \
  -d "${BC_DATASET}" \
  -o "${BC_OUT_DIR}" \
  --steps "${BC_STEPS:-200000}" \
  --batch_size "${BC_BATCH_SIZE:-512}" \
  --learning_rate "${BC_LR:-1e-4}" \
  --actor_hidden_dims "[512, 512, 512]" \
  --activation relu \
  --sampling_steps 10 \
  --cfm_loss_use_huber

