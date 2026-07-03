#!/bin/bash
#SBATCH -p day
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J rm75_mustard
#SBATCH -o /data/group/zhaolab/home/z2zhang/logs/%x_%j.out

set -euo pipefail

cd /data/group/zhaolab/home/z2zhang/vividex_sapien_server_upload_fpo_control_full

source /data/group/zhaolab/project/miniconda/etc/profile.d/conda.sh
conda activate /data/group/zhaolab/home/z2zhang/envs/vividex

export VIVIDEX_CONDA_ENV=/data/group/zhaolab/home/z2zhang/envs/vividex
export VIVIDEX_RUNTIME_ROOT=/data/group/zhaolab/home/z2zhang/results/vividex_rm75_runtime
export VIVIDEX_REPO_ROOT="$(pwd)"
export WANDB_MODE=offline
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

SEQ=ycb-006_mustard_bottle-20200709-subject-01-20200709_143211

mkdir -p /data/group/zhaolab/home/z2zhang/logs
mkdir -p "${VIVIDEX_RUNTIME_ROOT}"

echo "[job] host=$(hostname)"
echo "[job] SLURM_JOB_ID=${SLURM_JOB_ID}"
nvidia-smi || true
which python
python -V

python tools/retarget_rm75_inspire_reference.py \
  --src "norm_trajectories/${SEQ}.npz" \
  --dst "norm_trajectories/rm75_inspire_right/${SEQ}.npz" \
  --overwrite

bash scripts/server/train_state_single.sh "${SEQ}" rm75_inspire_mustard_fpo \
  "agent=fpo" \
  "env.robot_name=rm75_inspire_right" \
  "n_envs=8" \
  "n_eval_envs=2" \
  "total_timesteps=2000000" \
  "eval_freq=20000" \
  "eval_n_episodes=25" \
  "save_freq=50000" \
  "restore_checkpoint_freq=50000" \
  "agent.params.n_steps=4096" \
  "agent.params.batch_size=256" \
  "agent.params.n_epochs=5" \
  "agent.params.sampling_steps=8" \
  "agent.params.n_samples_per_action=16"

