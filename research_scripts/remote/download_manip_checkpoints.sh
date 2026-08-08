#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
mkdir -p "$CHECKPOINT_ROOT"

python -m pip install gdown
gdown --folder \
  https://drive.google.com/drive/folders/1vQ3Tv-mwNZIFipp5Bv0SQlfYfIhlf8_t \
  -O "$CHECKPOINT_ROOT" \
  --remaining-ok
