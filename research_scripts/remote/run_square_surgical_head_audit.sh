#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDITION="${CONDITION:-}"
case "$CONDITION" in
  control) ACTOR_TRAINABLE_SCOPE=all ;;
  candidate) ACTOR_TRAINABLE_SCOPE=output_head ;;
  *) echo "Set CONDITION to control or candidate" >&2; exit 1 ;;
esac

export NUM_ENVS=16
export DATA_COLLECTION_STEPS=320
# The loop stopping condition counts raw vectorized environment interactions.
export TOTAL_TIMESTEPS=10240
export EVAL_EPISODES=1
export ROLLOUT_FREQ=None
export ZERO_ENDPOINT_PCGRAD_TRAIN=False
export ROLLOUT_LOCAL_ACTOR_OPTIMIZER=False
export ADVANTAGE_SIGN_STRATIFIED_MINIBATCHES=False
export POTENTIAL_STAGE_SHAPING=False
export COLLECTION_FINGERPRINT_AUDIT=True
export SEED=20260981
export MASTER_PORT="${MASTER_PORT:-29500}"
export RUN_NAME="square_h52_${CONDITION}_audit_v4_osmesa_seed${SEED}"

export EXTRA_TRAIN_ARGS="--actor-trainable-scope $ACTOR_TRAINABLE_SCOPE --heldout-ratio-early-stop-audit True --heldout-ratio-early-stop-audit-iteration 2 --heldout-ratio-early-stop-audit-chunks 64"

"$SCRIPT_DIR/run_square_finetune.sh"
