# source ./thirdparty/miniconda3/bin/activate fpo_manipulation
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Create overall workspace
WORKSPACE_DIR=$SCRIPT_DIR/thirdparty
# Must match setup_env.sh: conda lives on local disk, not /workspace (bosfs).
# Do not honor ambient CONDA_ROOT (IsaacLab setup may export a different one).
CONDA_ROOT=$HOME/miniconda3_fpo_manipulation
ENV_ROOT=$CONDA_ROOT/envs/fpo_manipulation

eval "$($CONDA_ROOT/bin/conda shell.bash hook)"
conda activate $ENV_ROOT