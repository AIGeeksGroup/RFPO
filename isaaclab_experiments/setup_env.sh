# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Initialize git submodules (IsaacLab) only if missing.
# Skip when already vendored (bosfs/git-submodule clone often fails here).
if [[ ! -f "$SCRIPT_DIR/thirdparty/IsaacLab/isaaclab.sh" ]]; then
  if git rev-parse --git-dir > /dev/null 2>&1; then
    git submodule update --init --recursive
  fi
fi

# Create overall workspace
WORKSPACE_DIR=$SCRIPT_DIR/thirdparty
# Conda must live on a local executable filesystem. /workspace is bosfs (noexec);
# /root overlay is too small (~9GB free). Default: 30GB tmpfs (see setup_env.sh).
CONDA_ROOT=${CONDA_ROOT:-/tmp/isaaclab_conda/miniconda3_isaaclab_fpo}
ENV_ROOT=$CONDA_ROOT/envs/isaaclab_fpo
SENTINEL_FILE=.env_setup_finished
# Miniconda / conda / pip mirrors (override with empty string to use defaults).
MINICONDA_MIRROR=${MINICONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda}
CONDA_MIRROR=${CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda}
PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/workspace/.cache/pip}
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# Writable + executable space for conda/IsaacSim (bosfs /workspace cannot exec binaries).
ISAAC_CONDA_TMPFS=${ISAAC_CONDA_TMPFS:-/tmp/isaaclab_conda}
if [[ "$CONDA_ROOT" == ${ISAAC_CONDA_TMPFS}/* ]] && ! mountpoint -q "$ISAAC_CONDA_TMPFS"; then
  mkdir -p "$ISAAC_CONDA_TMPFS"
  mount -t tmpfs -o size=30G tmpfs "$ISAAC_CONDA_TMPFS"
fi

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  if [[ "$(lsb_release -is)" == "Ubuntu" ]]; then
    echo "skip apt: build-essential already present"
  fi

  # Install miniconda (prefer domestic mirror; official repo is slow from CN).
  # Download to local /tmp first — bosfs (/workspace) write is very slow for ~200MB files.
  if [[ ! -x $CONDA_ROOT/bin/conda ]]; then
    mkdir -p $CONDA_ROOT
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    if [[ ! -f "$MINICONDA_INSTALLER" ]] || [[ $(stat -c%s "$MINICONDA_INSTALLER" 2>/dev/null || echo 0) -lt 190000000 ]]; then
      rm -f "$MINICONDA_INSTALLER"
      curl -L --retry 3 --connect-timeout 15 \
        "${MINICONDA_MIRROR}/Miniconda3-latest-Linux-x86_64.sh" \
        -o "$MINICONDA_INSTALLER"
    fi
    bash "$MINICONDA_INSTALLER" -b -u -p $CONDA_ROOT
    rm -f "$MINICONDA_INSTALLER"
    $CONDA_ROOT/bin/conda config --system --set show_channel_urls yes
    $CONDA_ROOT/bin/conda config --system --remove-key channels 2>/dev/null || true
    $CONDA_ROOT/bin/conda config --system --add channels "${CONDA_MIRROR}/pkgs/main"
    $CONDA_ROOT/bin/conda config --system --add channels "${CONDA_MIRROR}/pkgs/r"
    $CONDA_ROOT/bin/conda config --system --set custom_channels.conda-forge "${CONDA_MIRROR}/cloud"
    $CONDA_ROOT/bin/conda config --system --set custom_channels.pytorch "${CONDA_MIRROR}/cloud"
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    # Prefer conda create; mamba on newest miniconda can hit python pin conflicts.
    $CONDA_ROOT/bin/conda create -y -n isaaclab_fpo python=3.10
  fi

  source $CONDA_ROOT/bin/activate isaaclab_fpo

  pip install -i "$PIP_INDEX" "numpy==1.26.4"
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  # setuptools<71 keeps pkg_resources, needed by flatdict==4.0.1's setup.py
  pip install -i "$PIP_INDEX" --upgrade pip "setuptools<71"

  pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

  # Pre-install flatdict==4.0.1 (source-only, uses pkg_resources which is
  # missing in pip's build-isolation with modern setuptools).
  pip install -i "$PIP_INDEX" --no-build-isolation "flatdict==4.0.1"
  bash thirdparty/IsaacLab/isaaclab.sh --install rsl_rl

  # isaaclab.sh may fail to install the core 'isaaclab' package (flatdict
  # version conflict during the find loop). Re-install it explicitly.
  # --no-deps avoids downgrading packages isaacsim already installed at
  # specific versions (onnx, prettytable, pillow, etc.)
  pip install -i "$PIP_INDEX" toml prettytable
  pip install -i "$PIP_INDEX" --no-deps --no-build-isolation --editable thirdparty/IsaacLab/source/isaaclab

  # Misc dependencies
  pip install -i "$PIP_INDEX" "opencv-python==4.9.0.80" "numba==0.61.2" \
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
