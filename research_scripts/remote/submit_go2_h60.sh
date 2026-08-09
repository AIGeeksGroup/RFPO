#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

STAGE="${1:-geometry}"
case "$STAGE" in
  geometry|control|candidate) ;;
  *)
    echo "Unknown stage: $STAGE (expected geometry, control, or candidate)" >&2
    exit 2
    ;;
esac

if [[ "$STAGE" != "geometry" ]]; then
  GEOMETRY_JSON="$RUNTIME_ROOT/results/go2_h60_equal_nfe_midpoint/geometry.json"
  python - "$GEOMETRY_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    geometry = json.load(handle)
if not geometry.get("passed", False):
    raise SystemExit("H60 Stage A did not pass; refusing to submit reward evaluation")
PY
fi

TASK_NAME="go2-h60-$STAGE"
TASK_LOG="$RUNTIME_ROOT/logs/${TASK_NAME}_htrain.log"
TASK_CMD="FPO_PROJECT_ROOT=$PROJECT_ROOT FPO_RUNTIME_ROOT=$RUNTIME_ROOT PHYSICAL_GPU=0 STAGE=$STAGE bash $SCRIPT_DIR/run_go2_h60.sh"

exec /usr/local/bin/submit \
  --name "$TASK_NAME" \
  --nodes 1 \
  --gpus-per-node 1 \
  --project general \
  --log-path "$TASK_LOG" \
  --cmd "$TASK_CMD"
