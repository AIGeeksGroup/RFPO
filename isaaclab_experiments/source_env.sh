SOURCE_ENV_SETUP_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ -n "${ZSH_VERSION:-}" ]; then
    SOURCE_ENV_SETUP_DIR="$( cd "$( dirname "$0" )" && pwd )"
fi
source ${SOURCE_ENV_SETUP_DIR}/thirdparty/miniconda3/bin/activate isaaclab_fpo

# # Environment variable hacks: these are only needed when installing IsaacSim without pip, from the release zip.
# # Set PYTHONPATH env variable within the conda environment.
# export BACKUP_PYTHONPATH=$PYTHONPATH
# source ${SOURCE_ENV_SETUP_DIR}/thirdparty/IsaacLab/_isaac_sim/setup_conda_env.sh
# conda env config vars set PYTHONPATH="$PYTHONPATH"
# export PYTHONPATH=$BACKUP_PYTHONPATH
# # Need to deactivate => reactivate for the variable to take effect.
# conda deactivate
# source ${SOURCE_ENV_SETUP_DIR}/thirdparty/miniconda3/bin/activate isaaclab_fpo

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SOURCE_ENV_SETUP_DIR}/thirdparty/miniconda3/envs/isaaclab_fpo/lib
export OMNI_KIT_ACCEPT_EULA=YES

# Workaround: newer NVIDIA open kernel modules (driver 580+, kernel 6.17+) may
# not auto-enumerate GPUs for CUDA. nvidia-smi works but cuInit() fails without this.
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi
