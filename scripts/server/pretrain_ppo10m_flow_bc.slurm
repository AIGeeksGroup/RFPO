#!/bin/bash
#SBATCH -p day
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J pretrain_ppo10m_bc
#SBATCH -o /data/group/zhaolab/home/z2zhang/logs/%x_%j.out

set -euo pipefail

export VIVIDEX_REPO_ROOT="${VIVIDEX_REPO_ROOT:-/data/group/zhaolab/home/z2zhang/vividex_sapien_server_upload_fpo_control_full}"
export VIVIDEX_CONDA_ENV="${VIVIDEX_CONDA_ENV:-/data/group/zhaolab/home/z2zhang/envs/vividex}"
export VIVIDEX_CONDA_SH="${VIVIDEX_CONDA_SH:-/data/group/zhaolab/project/miniconda/etc/profile.d/conda.sh}"
export VIVIDEX_RUNTIME_ROOT="${VIVIDEX_RUNTIME_ROOT:-/data/group/zhaolab/home/z2zhang/results/vividex_fpo_runtime}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p /data/group/zhaolab/home/z2zhang/logs
cd "${VIVIDEX_REPO_ROOT}"

# shellcheck disable=SC1091
source scripts/server/common.sh
vividex_setup_runtime
vividex_activate_env
vividex_print_context
nvidia-smi || true

DATASET="${BC_DATASET:-/data/group/zhaolab/home/z2zhang/data/fpo_bc/mustard_ppo10m_stage2.npz}"
OUT_DIR="${BC_OUT_DIR:-/data/group/zhaolab/home/z2zhang/results/fpo_bc_mustard_ppo10m}"

python tools/pretrain_fpo_bc.py \
  -d "${DATASET}" \
  -o "${OUT_DIR}" \
  --steps 200000 \
  --batch_size 512 \
  --learning_rate 1e-4 \
  --actor_hidden_dims "[512, 512, 512]" \
  --activation relu \
  --sampling_steps 10 \
  --cfm_loss_use_huber
