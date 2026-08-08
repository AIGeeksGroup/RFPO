# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Initialize git submodules (IsaacLab)
# Skip if not in a git repo (e.g., SkyPilot workdir sync strips .git)
if git rev-parse --git-dir > /dev/null 2>&1; then
  git submodule update --init --recursive
fi

# Create overall workspace
WORKSPACE_DIR=$SCRIPT_DIR/thirdparty
CONDA_ROOT=$WORKSPACE_DIR/miniconda3
ENV_ROOT=$CONDA_ROOT/envs/isaaclab_fpo
SENTINEL_FILE=.env_setup_finished

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  if [[ "$(lsb_release -is)" == "Ubuntu" ]] && ! command -v gcc >/dev/null 2>&1; then
    sudo apt install -y build-essential
  fi

  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n isaaclab_fpo python=3.10
  fi

  # Isaac Sim's Iray plugin links against libGLU, which is not installed on
  # minimal headless servers. Keep it inside the project environment.
  $CONDA_ROOT/bin/conda install -y -n isaaclab_fpo -c conda-forge libglu

  source $CONDA_ROOT/bin/activate isaaclab_fpo

  pip install "numpy==1.26.4"
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  # setuptools<71 keeps pkg_resources, needed by flatdict==4.0.1's setup.py
  pip install --upgrade pip "setuptools<71"

  pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

  # Pre-install flatdict==4.0.1 (source-only, uses pkg_resources which is
  # missing in pip's build-isolation with modern setuptools).
  pip install --no-build-isolation "flatdict==4.0.1"
  bash thirdparty/IsaacLab/isaaclab.sh --install rsl_rl

  # isaaclab.sh may fail to install the core 'isaaclab' package (flatdict
  # version conflict during the find loop). Re-install it explicitly.
  # --no-deps avoids downgrading packages isaacsim already installed at
  # specific versions (onnx, prettytable, pillow, etc.)
  pip install toml prettytable
  pip install --no-deps --no-build-isolation --editable thirdparty/IsaacLab/source/isaaclab

  # Misc dependencies
  pip install "opencv-python==4.9.0.80" "numba==0.61.2" \
    "websockets==15.0.1" "wandb==0.25.1" "viser==1.0.24"

  # Download robot description files for whole_body_tracking
  WBT_ASSETS="$SCRIPT_DIR/thirdparty/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets"
  if [ ! -d "$WBT_ASSETS/unitree_description" ]; then
    curl -L -o /tmp/unitree_description.tar.gz \
      https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
    tar -xzf /tmp/unitree_description.tar.gz -C "$WBT_ASSETS/"
    rm /tmp/unitree_description.tar.gz
  fi

  # Our packages
  pip install -e ./isaaclab_fpo \
    -e ./thirdparty/whole_body_tracking/source/whole_body_tracking

  # Download LAFAN1 motion data and convert to NPZ via IsaacSim FK
  export OMNI_KIT_ACCEPT_EULA=YES
  python $SCRIPT_DIR/whole_body_tracking_reference_data/download_lafan_data.py --headless

  touch $SENTINEL_FILE
fi

export OMNI_KIT_ACCEPT_EULA=YES
