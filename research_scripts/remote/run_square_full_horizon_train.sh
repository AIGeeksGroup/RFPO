#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NUM_ENVS=16
export DATA_COLLECTION_STEPS=400
export TOTAL_TIMESTEPS=25600
export EVAL_EPISODES=1
export ROLLOUT_FREQ=None
export ZERO_ENDPOINT_PCGRAD_TRAIN=False
export ROLLOUT_LOCAL_ACTOR_OPTIMIZER=False
export ADVANTAGE_SIGN_STRATIFIED_MINIBATCHES=False
export SEED="${SEED:-20260916}"
export RUN_NAME="${RUN_NAME:-square_h47_fullhorizon_osmesa_seed${SEED}}"

"$SCRIPT_DIR/run_square_finetune.sh"
