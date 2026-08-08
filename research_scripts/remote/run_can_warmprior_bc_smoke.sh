#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
CAN_CHECKPOINT="$CHECKPOINT_ROOT/95j3noe4_step_1000"
RUN_TAG="${RUN_TAG:-can_warmprior_bc_smoke_$(timestamp)}"

if [[ ! -f "$CAN_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Can checkpoint: $CAN_CHECKPOINT" >&2
  exit 1
fi

python pretrain_flow_bc.py \
  --dataset ankile/robomimic-mh-can-image \
  --resume-ckpt "$CAN_CHECKPOINT" \
  --steps 1002 \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --source-prior-mode previous_action \
  --source-prior-sigma "${SOURCE_PRIOR_SIGMA:-0.5}" \
  --log-freq 1 \
  --save-freq 10000 \
  --wandb-enable False \
  --output-dir "$RUNTIME_ROOT/results/$RUN_TAG"
