#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PHYSICAL_GPU="${PHYSICAL_GPU:-0}"
STAGE="${STAGE:-geometry}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STABLE_POLLS="${STABLE_POLLS:-4}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"

case "$STAGE" in
  geometry|control|candidate) ;;
  *)
    echo "Unknown stage: $STAGE (expected geometry, control, or candidate)" >&2
    exit 2
    ;;
esac

GPU_UUID="$({ nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits; } \
  | awk -F ', ' -v gpu_index="$PHYSICAL_GPU" '$1 == gpu_index { print $2 }')"
if [[ -z "$GPU_UUID" ]]; then
  echo "Could not resolve physical GPU index $PHYSICAL_GPU" >&2
  exit 2
fi

started_at=$SECONDS
free_polls=0
echo "Waiting for GPU $PHYSICAL_GPU ($GPU_UUID) to remain free for $STABLE_POLLS polls"
while ((SECONDS - started_at < MAX_WAIT_SECONDS)); do
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -Fx "$GPU_UUID" >/dev/null; then
    free_polls=0
  else
    ((free_polls += 1))
    echo "$(date -Is) GPU $PHYSICAL_GPU free poll $free_polls/$STABLE_POLLS"
    if ((free_polls >= STABLE_POLLS)); then
      echo "$(date -Is) launching H60 stage $STAGE on GPU $PHYSICAL_GPU"
      exec env PHYSICAL_GPU="$PHYSICAL_GPU" STAGE="$STAGE" \
        bash "$SCRIPT_DIR/run_go2_h60.sh"
    fi
  fi
  sleep "$POLL_SECONDS"
done

echo "Timed out after $MAX_WAIT_SECONDS seconds waiting for GPU $PHYSICAL_GPU" >&2
exit 1
