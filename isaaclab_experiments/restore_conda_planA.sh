#!/usr/bin/env bash
# After reboot/shm wipe: extract workspace archive to shm, then root symlink.
set -euo pipefail
ARCH=${ARCH:-/workspace/plsy/miniconda3_isaaclab_fpo.tar.zst}
ZSTD=${ZSTD:-/usr/bin/zstd}
STAGING=/dev/shm/_isaaclab_staging
LINK=/root/miniconda3_isaaclab_fpo
if [[ -x "$LINK/envs/isaaclab_fpo/bin/python" ]]; then
  echo "already present: $LINK"
  exit 0
fi
[[ -f "$ARCH" ]] || { echo "missing $ARCH" >&2; exit 1; }
[[ -x "$ZSTD" ]] || { echo "need zstd at $ZSTD (apt install zstd)" >&2; exit 1; }
mkdir -p "$STAGING"
"$ZSTD" -dc "$ARCH" | tar -C "$STAGING" -xf -
ln -sfn "$STAGING/miniconda3" "$LINK"
echo "restored $LINK -> $STAGING/miniconda3"
