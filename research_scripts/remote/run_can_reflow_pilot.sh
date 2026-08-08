#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

MODE="${1:-}"
if [[ "$MODE" != "control" && "$MODE" != "reflow" ]]; then
  echo "Usage: $0 {control|reflow}" >&2
  exit 2
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
CAN_CHECKPOINT="$CHECKPOINT_ROOT/95j3noe4_step_1000"
RUN_TAG="${RUN_TAG:-can_${MODE}100_seed20260808_$(timestamp)}"

extra_args=()
if [[ "$MODE" == "reflow" ]]; then
  extra_args+=(
    --reflow-teacher-ckpt "$CAN_CHECKPOINT"
    --reflow-teacher-sampling-steps "${TEACHER_STEPS:-64}"
  )
fi

python pretrain_flow_bc.py \
  --dataset ankile/robomimic-mh-can-image \
  --resume-ckpt "$CAN_CHECKPOINT" \
  --load-ema True \
  --restart-training-state True \
  --steps "${UPDATES:-100}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --seed "${TRAIN_SEED:-20260808}" \
  --log-freq "${LOG_FREQ:-10}" \
  --save-freq 10000 \
  --wandb-enable False \
  --output-dir "$RUNTIME_ROOT/results/$RUN_TAG" \
  "${extra_args[@]}"

