#!/bin/bash
#SBATCH -p day
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J collect_ppo10m_bc
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

PPO_RUN="${PPO_RUN:-/data/group/zhaolab/home/z2zhang/results/vividex/results/state_baseline/ppo_mustard_baseline}"
PPO_10M="${PPO_10M:-${PPO_RUN}/logs/rl_models_10000000_steps.zip}"
OUT="${BC_DATASET:-/data/group/zhaolab/home/z2zhang/data/fpo_bc/mustard_ppo10m_stage2.npz}"
NUM_TRAJS="${NUM_TRAJS:-100}"

python tools/collect_fpo_bc_dataset.py \
  --checkpoint_file "${PPO_10M}" \
  --config_dir "${PPO_RUN}" \
  -o "${OUT}" \
  -n "${NUM_TRAJS}" \
  --stage 2 \
  --success_metric sr10 \
  --norm_traj
