#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDITION="${CONDITION:-}"
if [[ "$CONDITION" != "control" && "$CONDITION" != "candidate" ]]; then
  echo "Set CONDITION to control or candidate" >&2
  exit 1
fi

export NUM_ENVS=16
export DATA_COLLECTION_STEPS=320
export TOTAL_TIMESTEPS=25600
export EVAL_EPISODES=1
export ROLLOUT_FREQ=None
export ZERO_ENDPOINT_PCGRAD_TRAIN=False
export ROLLOUT_LOCAL_ACTOR_OPTIMIZER=False
export ADVANTAGE_SIGN_STRATIFIED_MINIBATCHES=False
export POTENTIAL_STAGE_SHAPING=False
export COLLECTION_FINGERPRINT_AUDIT=True
if [[ "$CONDITION" == "candidate" ]]; then
  export POTENTIAL_STAGE_SHAPING=True
fi
export SEED=20260977
export MASTER_PORT="${MASTER_PORT:-29500}"
export RUN_NAME="square_h51_confirmation_${CONDITION}_osmesa_seed${SEED}"

"$SCRIPT_DIR/run_square_finetune.sh"

