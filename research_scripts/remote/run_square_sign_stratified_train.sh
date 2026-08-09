#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

RUN_NAME="${RUN_NAME:-square_h45_sign_stratified_osmesa_seed20260916}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNTIME_ROOT/results/$RUN_NAME}"
LOG_PATH="${LOG_PATH:-$RUNTIME_ROOT/logs/$RUN_NAME.log}"
if [[ -e "$OUTPUT_DIR" || -e "$LOG_PATH" ]]; then
  echo "Refusing to overwrite existing H45 artifact: $RUN_NAME" >&2
  exit 1
fi

RUN_NAME="$RUN_NAME" \
OUTPUT_DIR="$OUTPUT_DIR" \
LOG_PATH="$LOG_PATH" \
SEED=20260916 \
MASTER_PORT="${MASTER_PORT:-29545}" \
ADVANTAGE_SIGN_STRATIFIED_MINIBATCHES=True \
bash "$SCRIPT_DIR/run_square_finetune.sh"
