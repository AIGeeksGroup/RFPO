#!/bin/bash
#SBATCH -A willow
#SBATCH --job-name=ho_sim            # job name
#SBATCH --cpus-per-gpu=14           # number of cores per tasks
#SBATCH --partition=willow          # number of GPUs per node
#SBATCH --gres=gpu:1                # number of GPUs per node
#SBATCH --mem=40G                   # number of memory allocated
#SBATCH --time=39:59:59              # maximum execution time (HH:MM:SS)
#SBATCH --output=./outputs/output_logs/%j.out # output file name
#SBATCH --error=./outputs/output_logs/%j.err  # error file name
#SBATCH --signal=USR1@20

module purge
# module load cuda/10.2.89

# activate anaconda
eval "$(conda shell.bash hook)"
conda activate cpu
wandb offline

srun python tools/imitate_train.py ${@}
