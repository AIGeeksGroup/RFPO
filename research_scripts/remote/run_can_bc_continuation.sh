#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

PRIOR_MODE="${1:-}"
if [[ "$PRIOR_MODE" != "gaussian" && "$PRIOR_MODE" != "previous_action" ]]; then
  echo "Usage: $0 {gaussian|previous_action}" >&2
  exit 2
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
CAN_CHECKPOINT="$CHECKPOINT_ROOT/95j3noe4_step_1000"
RUN_TAG="${RUN_TAG:-can_bc100_${PRIOR_MODE}_seed20260808_$(timestamp)}"

if [[ ! -f "$CAN_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Can checkpoint: $CAN_CHECKPOINT" >&2
  exit 1
fi

python pretrain_flow_bc.py \
  --dataset ankile/robomimic-mh-can-image \
  --resume-ckpt "$CAN_CHECKPOINT" \
  --steps "${END_STEP:-1101}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --seed "${TRAIN_SEED:-20260808}" \
  --source-prior-mode "$PRIOR_MODE" \
  --source-prior-sigma "${SOURCE_PRIOR_SIGMA:-0.5}" \
  --log-freq "${LOG_FREQ:-10}" \
  --save-freq 10000 \
  --wandb-enable False \
  --output-dir "$RUNTIME_ROOT/results/$RUN_TAG"

