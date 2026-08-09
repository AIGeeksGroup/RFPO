#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDITION="${CONDITION:-}"
if [[ "$CONDITION" != "control" && "$CONDITION" != "candidate" ]]; then
  echo "Set CONDITION to control or candidate" >&2
  exit 1
fi

export NUM_ENVS="${NUM_ENVS:-16}"
export DATA_COLLECTION_STEPS="${DATA_COLLECTION_STEPS:-320}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-25600}"
export EVAL_EPISODES=1
export ROLLOUT_FREQ=None
export ZERO_ENDPOINT_PCGRAD_TRAIN=False
if [[ "$CONDITION" == "candidate" ]]; then
  export ZERO_ENDPOINT_PCGRAD_TRAIN=True
fi
export SEED="${SEED:-20260913}"
export RUN_NAME="${RUN_NAME:-square_h41_${CONDITION}_osmesa_seed${SEED}}"

"$SCRIPT_DIR/run_square_finetune.sh"
