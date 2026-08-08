# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Create overall workspace
WORKSPACE_DIR=$SCRIPT_DIR/thirdparty
CONDA_ROOT=$WORKSPACE_DIR/miniconda3
ENV_ROOT=$CONDA_ROOT/envs/fpo_manipulation
SENTINEL_FILE=.env_setup_finish

mkdir -p $WORKSPACE_DIR

# The vendored robosuite 1.5.1 tree omits its binary texture directory.
# Fetch only those pinned upstream assets while preserving local robot additions.
ROBOSUITE_TEXTURE_SENTINEL=$WORKSPACE_DIR/robosuite/robosuite/models/assets/textures/light-gray-floor-tile.png
if [[ ! -f $ROBOSUITE_TEXTURE_SENTINEL ]]; then
  ROBOSUITE_ASSET_TMP=$(mktemp -d)
  trap 'rm -rf "$ROBOSUITE_ASSET_TMP"' EXIT
  curl --fail --location \
    https://github.com/ARISE-Initiative/robosuite/archive/refs/tags/v1.5.1.tar.gz \
    -o "$ROBOSUITE_ASSET_TMP/robosuite-v1.5.1.tar.gz"
  tar -xzf "$ROBOSUITE_ASSET_TMP/robosuite-v1.5.1.tar.gz" \
    -C "$ROBOSUITE_ASSET_TMP"
  cp -a "$ROBOSUITE_ASSET_TMP/robosuite-1.5.1/robosuite/models/assets/textures" \
    "$WORKSPACE_DIR/robosuite/robosuite/models/assets/"
  rm -rf "$ROBOSUITE_ASSET_TMP"
  trap - EXIT
fi

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then

    $CONDA_ROOT/bin/conda tos accept
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n fpo_manipulation python=3.10

  fi

  # Initialize conda and activate environment
  eval "$($CONDA_ROOT/bin/conda shell.bash hook)"
  conda activate fpo_manipulation

  # Ensure gymnasium 1.1.1 is installed
  python -m pip install gymnasium==1.1.1

  python -m pip install -e $WORKSPACE_DIR/robosuite --no-deps

  # Weird... it's causing the error in the server by referencing the base environment
  # python $WORKSPACE_DIR/robosuite/robosuite/scripts/setup_macros.py # avoid the warning

  # Git clone dexmimicgen into WORKSPACE_DIR/dexmimicgen if not already present
  # Note: git archive creates an empty dir for gitlink submodules, so check for
  # setup.py inside rather than just the directory's existence.
  if [ ! -f $WORKSPACE_DIR/dexmimicgen/setup.py ]; then
    rm -rf $WORKSPACE_DIR/dexmimicgen
    git clone https://github.com/NVlabs/dexmimicgen.git $WORKSPACE_DIR/dexmimicgen
  fi

  # Install dexmimicgen
  python -m pip install -e $WORKSPACE_DIR/dexmimicgen

  # Install PyOpenGL-accelerate
  python -m pip install PyOpenGL-accelerate

  cd $WORKSPACE_DIR

  # Install ffmpeg and pkg-config FIRST (before lerobot requirements)
  # conda install --override-channels -c conda-forge ffmpeg==4.4.2 pkg-config -y
  conda install -c conda-forge ffmpeg=7.1.1 pkg-config -y
  
  # Install a couple of dependencies
  python -m pip install -r $WORKSPACE_DIR/lerobot_requirements.txt --no-deps
  python -m pip install transformers==4.46.3
  python -m pip install -e $WORKSPACE_DIR/lerobot --no-deps 

  python -m pip install matplotlib
  python -m pip install seaborn
  python -m pip install tyro

  cd ..

  touch $SENTINEL_FILE

fi
