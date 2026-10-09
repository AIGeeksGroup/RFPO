#!/usr/bin/env bash
# Resume setup_env.sh after isaacsim + flatdict, then Plan A migrate.
set -euo pipefail
EXP_DIR=/workspace/plsy/fgo_test/isaaclab_experiments
LOG="${EXP_DIR}/setup_planA.log"
exec > >(tee -a "$LOG") 2>&1
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

STAGING=/dev/shm/_isaaclab_staging
CONDA_ROOT="${STAGING}/miniconda3"
TMP=/dev/shm/_isaaclab_tmp
DEST=/workspace/plsy/miniconda3_isaaclab_fpo
LINK=/root/miniconda3_isaaclab_fpo

export CONDA_ROOT
export CONDA_PKGS_DIRS=/dev/shm/_isaaclab_pkgs
export TMPDIR="$TMP"
# bosfs pip wheel cache breaks setuptools; keep cache on shm
export PIP_CACHE_DIR="${TMP}/pip-cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$TMP" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

# shellcheck disable=SC1091
source "$CONDA_ROOT/bin/activate" isaaclab_fpo
cd "$EXP_DIR"

log "=== resume: isaaclab.sh --install rsl_rl ==="
# isaaclab.sh has set -e; some extension installs may fail (documented). Don't abort wrapper.
set +e
bash thirdparty/IsaacLab/isaaclab.sh --install rsl_rl
rc=$?
set -e
log "isaaclab.sh exit=$rc"

log "=== resume: explicit isaaclab + extras ==="
pip install --no-cache-dir toml prettytable
pip install --no-deps --no-build-isolation --no-cache-dir --editable thirdparty/IsaacLab/source/isaaclab

pip install --no-cache-dir "opencv-python==4.9.0.80" "numba==0.61.2" \
  "websockets==15.0.1" "wandb==0.25.1" "viser==1.0.24"

WBT_ASSETS="$EXP_DIR/thirdparty/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets"
if [ ! -d "$WBT_ASSETS/unitree_description" ]; then
  log "download unitree_description"
  curl -L -o "$TMP/unitree_description.tar.gz" \
    https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
  tar -xzf "$TMP/unitree_description.tar.gz" -C "$WBT_ASSETS/"
  rm -f "$TMP/unitree_description.tar.gz"
fi

log "=== resume: isaaclab_fpo + whole_body_tracking ==="
pip install --no-cache-dir -e ./isaaclab_fpo \
  -e ./thirdparty/whole_body_tracking/source/whole_body_tracking

log "=== resume: LAFAN npz (IsaacSim headless) ==="
set +e
python "$EXP_DIR/whole_body_tracking_reference_data/download_lafan_data.py" --headless
lafan_rc=$?
set -e
log "lafan exit=$lafan_rc"

touch "$EXP_DIR/.env_setup_finished"
log "sentinel written"

log "=== Plan A migrate: tar deref to workspace ==="
"$CONDA_ROOT/bin/conda" clean -afy || true
rm -rf "$TMP" /dev/shm/_isaaclab_src /dev/shm/_isaaclab_pkgs
rm -f "$CONDA_ROOT/miniconda.sh" || true
log "disk before migrate"
df -h / /dev/shm /workspace | sed 's/^/[df] /'
du -sh "$CONDA_ROOT" | sed 's/^/[du] /'

rm -rf "$DEST"
mkdir -p "$DEST"
log "tar --dereference --hard-dereference -> $DEST"
tar -C "$CONDA_ROOT" --dereference --hard-dereference -cf - . | tar -xf - -C "$DEST"
rm -rf "$LINK"
ln -sfn "$DEST" "$LINK"
ls -ld "$LINK"
rm -rf "$STAGING"

log "=== smoke ==="
# shellcheck disable=SC1091
source "$LINK/bin/activate" isaaclab_fpo
python -c 'import sys; print("env", sys.executable, sys.version)'
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
python -c 'import isaaclab; print("isaaclab", isaaclab.__file__)'
python -c 'import isaaclab_fpo; print("isaaclab_fpo", isaaclab_fpo.__file__)'
log "PLAN A DONE (lafan_rc=$lafan_rc)"
df -h / /dev/shm /workspace | sed 's/^/[df] /'
du -sh "$DEST" | sed 's/^/[du dest] /'
