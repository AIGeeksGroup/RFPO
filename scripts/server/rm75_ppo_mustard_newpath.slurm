#!/bin/bash
#SBATCH -p day
#SBATCH -t 23:59:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J rm75_ppo
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
RUN_NAME=rm75_ppo_mustard_newpath
RUN_DIR="${VIVIDEX_RUNTIME_ROOT}/results/state_baseline/${RUN_NAME}"

mkdir -p /data/group/zhaolab/home/z2zhang/logs
mkdir -p "${VIVIDEX_RUNTIME_ROOT}" "${RUN_DIR}"

echo "[job] host=$(hostname)"
echo "[job] SLURM_JOB_ID=${SLURM_JOB_ID}"
nvidia-smi || true
which python
python -V

RM75_TRAJ="norm_trajectories/rm75_inspire_right/${SEQ}.npz"
if [ ! -f "${RM75_TRAJ}" ]; then
  python tools/retarget_rm75_inspire_reference.py \
    --src "norm_trajectories/${SEQ}.npz" \
    --dst "${RM75_TRAJ}" \
    --overwrite
else
  echo "[rm75-train] using existing retargeted trajectory: ${RM75_TRAJ}"
fi

python tools/train_rm75_sim.py \
  --seq-name "${SEQ}" \
  --output-dir "${RUN_DIR}" \
  --n-envs 8 \
  --n-eval-envs 2 \
  --total-timesteps 2000000 \
  --eval-freq 20000 \
  --eval-n-episodes 25 \
  --save-freq 50000 \
  --restore-checkpoint-freq 50000 \
  --norm-traj
