#!/bin/bash
#SBATCH -p day
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J rfpo_ppo10m_bc
#SBATCH -o /data/group/zhaolab/home/z2zhang/logs/%x_%j.out

set -euo pipefail

export VIVIDEX_REPO_ROOT="${VIVIDEX_REPO_ROOT:-/data/group/zhaolab/home/z2zhang/vividex_sapien_server_upload_fpo_control_full}"
export VIVIDEX_CONDA_ENV="${VIVIDEX_CONDA_ENV:-/data/group/zhaolab/home/z2zhang/envs/vividex}"
export VIVIDEX_CONDA_SH="${VIVIDEX_CONDA_SH:-/data/group/zhaolab/project/miniconda/etc/profile.d/conda.sh}"
export VIVIDEX_RUNTIME_ROOT="${VIVIDEX_RUNTIME_ROOT:-/data/group/zhaolab/home/z2zhang/results/vividex_fpo_runtime}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p /data/group/zhaolab/home/z2zhang/logs
cd "${VIVIDEX_REPO_ROOT}"

SEQ="${SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"
RUN_NAME="${RUN_NAME:-rfpo_mustard_ppo10m_bc_aspo_2m}"
DATASET="${BC_DATASET:-/data/group/zhaolab/home/z2zhang/data/fpo_bc/mustard_ppo10m_stage2.npz}"
BC_CKPT="${BC_CKPT:-/data/group/zhaolab/home/z2zhang/results/fpo_bc_mustard_ppo10m/flow_bc_step_200000.pt}"

bash scripts/server/train_state_single.sh "${SEQ}" "${RUN_NAME}" \
  "agent=fpo" \
  "agent.params.trust_region_mode=aspo" \
  "agent.params.bc_checkpoint=${BC_CKPT}" \
  "agent.params.bc_anchor_dataset=${DATASET}" \
  "agent.params.bc_anchor_coef=0.03" \
  "agent.params.bc_anchor_decay_steps=600000" \
  "agent.params.bc_anchor_batch_size=256" \
  "agent.params.learning_rate=3.0e-6" \
  "agent.params.max_grad_norm=0.5" \
  "agent.params.n_steps=4096" \
  "agent.params.batch_size=256" \
  "agent.params.n_epochs=1" \
  "agent.params.sampling_steps=8" \
  "agent.params.n_samples_per_action=16" \
  "agent.params.spo_clip_coef=0.005" \
  "n_envs=8" \
  "n_eval_envs=2" \
  "total_timesteps=2000000" \
  "eval_freq=20000" \
  "save_freq=50000" \
  "restore_checkpoint_freq=50000"
