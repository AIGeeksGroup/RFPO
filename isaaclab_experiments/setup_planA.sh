#!/usr/bin/env bash
# Plan A: stage on local tmpfs (chmod/symlink OK) -> tar to workspace -> root symlink.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${EXP_DIR}/setup_planA.log"
exec > >(tee -a "$LOG") 2>&1

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

STAGING=/dev/shm/_isaaclab_staging
SRC=/dev/shm/_isaaclab_src
TMP=/dev/shm/_isaaclab_tmp
PKGS=/dev/shm/_isaaclab_pkgs
CONDA_ROOT="${STAGING}/miniconda3"
DEST=/workspace/plsy/miniconda3_isaaclab_fpo
LINK=/root/miniconda3_isaaclab_fpo
ISAACLAB_SHA=21f7136325136ca3f6ca4e0a8125edffe5c24f7e
WBT_SHA=cd65172032893724b445448818c34165846d847d

export CONDA_ROOT
export CONDA_PKGS_DIRS="$PKGS"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export TMPDIR="$TMP"
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_TERMINAL_PROMPT=0
export PYTHONNOUSERSITE=1

mkdir -p "$STAGING" "$SRC" "$TMP" "$PKGS" "$PIP_CACHE_DIR"

log "disk before"
df -h / /dev/shm /workspace | sed 's/^/[df] /'

populate_thirdparty() {
  local name="$1" url="$2" sha="$3" dest="$4"
  if [[ -f "$dest/isaaclab.sh" || -f "$dest/source/whole_body_tracking/setup.py" || -f "$dest/setup.py" ]]; then
    log "$name already present at $dest"
    return 0
  fi
  if [[ ! -d "$SRC/$name/.git" ]]; then
    log "clone $name"
    git clone --filter=blob:none "$url" "$SRC/$name"
  fi
  log "checkout $name $sha"
  git -C "$SRC/$name" fetch --filter=blob:none origin "$sha"
  git -C "$SRC/$name" checkout --force "$sha"
  mkdir -p "$dest"
  log "copy $name -> $dest (deref, no hardlinks)"
  tar -C "$SRC/$name" --exclude='.git' --dereference --hard-dereference -cf - . \
    | tar -xf - -C "$dest"
}

log "=== 1) thirdparty sources ==="
populate_thirdparty IsaacLab https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_SHA" \
  "$EXP_DIR/thirdparty/IsaacLab"
populate_thirdparty whole_body_tracking https://github.com/HybridRobotics/whole_body_tracking.git "$WBT_SHA" \
  "$EXP_DIR/thirdparty/whole_body_tracking"

if [[ ! -f "$EXP_DIR/thirdparty/IsaacLab/isaaclab.sh" ]]; then
  log "ERROR: IsaacLab/isaaclab.sh missing after clone"
  exit 1
fi

log "=== 2) setup_env.sh on staging CONDA_ROOT=$CONDA_ROOT ==="
cd "$EXP_DIR"
# setup_env.sh uses set -ex; keep going from this wrapper only if it succeeds
bash setup_env.sh

log "=== 3) shrink staging then tar to workspace ==="
if [[ -x "$CONDA_ROOT/bin/conda" ]]; then
  "$CONDA_ROOT/bin/conda" clean -afy || true
fi
rm -rf "$TMP" "$PKGS" "$SRC"
# drop installer leftovers if any
rm -f "$CONDA_ROOT/miniconda.sh" || true

log "disk before migrate"
df -h / /dev/shm /workspace | sed 's/^/[df] /'
du -sh "$CONDA_ROOT" | sed 's/^/[du] /'

if [[ -e "$DEST" && ! -L "$DEST" ]]; then
  log "removing previous dest $DEST"
  rm -rf "$DEST"
fi
mkdir -p "$DEST"
log "tar --dereference --hard-dereference -> $DEST"
tar -C "$CONDA_ROOT" --dereference --hard-dereference -cf - . | tar -xf - -C "$DEST"

log "=== 4) root symlink ==="
# If LINK is a real dir, replace with symlink
if [[ -e "$LINK" || -L "$LINK" ]]; then
  rm -rf "$LINK"
fi
ln -sfn "$DEST" "$LINK"
ls -ld "$LINK"

log "=== 5) drop staging ==="
rm -rf "$STAGING"
log "disk after"
df -h / /dev/shm /workspace | sed 's/^/[df] /'

log "=== 6) smoke ==="
"$LINK/bin/python" -c 'import sys; print("base python", sys.executable, sys.version)'
# shellcheck disable=SC1091
source "$LINK/bin/activate" isaaclab_fpo
python -c 'import sys; print("env python", sys.executable, sys.version)'
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
python -c 'import isaaclab; print("isaaclab ok", getattr(isaaclab, "__file__", "?"))' || python -c 'import isaacsim; print("isaacsim ok")'
log "PLAN A DONE"
df -h / /dev/shm /workspace | sed 's/^/[df] /'
