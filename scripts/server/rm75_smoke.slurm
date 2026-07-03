#!/bin/bash
#SBATCH -p day
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J rm75_smoke
#SBATCH -o /data/group/zhaolab/home/z2zhang/logs/%x_%j.out

set -euo pipefail

cd /data/group/zhaolab/home/z2zhang/vividex_sapien_server_upload_fpo_control_full

source /data/group/zhaolab/project/miniconda/etc/profile.d/conda.sh
conda activate /data/group/zhaolab/home/z2zhang/envs/vividex

export VIVIDEX_CONDA_ENV=/data/group/zhaolab/home/z2zhang/envs/vividex
export VIVIDEX_RUNTIME_ROOT=/data/group/zhaolab/home/z2zhang/results/vividex_rm75_runtime
export VIVIDEX_REPO_ROOT="$(pwd)"
export VIVIDEX_ROBOT_NAME=rm75_inspire_right
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

bash scripts/server/smoke_test.sh "${SEQ}" rm75_inspire_smoke \
  "agent=fpo" \
  "total_timesteps=32"

