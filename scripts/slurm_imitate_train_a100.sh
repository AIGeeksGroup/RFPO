#!/bin/bash
#SBATCH -A ets@a100
#SBATCH -C a100
#SBATCH --job-name=ho_sim          # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=20           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=19:59:59              # maximum execution time (HH:MM:SS)
#SBATCH --output=./outputs/output_logs/%j.out # output file name
#SBATCH --error=./outputs/output_logs/%j.err  # error file name
#SBATCH --signal=USR1@20

#module load cuda/11.7.1 && module load cudnn/8.5.0.96-11.7-cuda
# activate anaconda
eval "$(conda shell.bash hook)"
conda activate cpu
wandb offline

export TMPDIR=$JOBSCRATCH
srun python tools/imitate_train.py ${@}
