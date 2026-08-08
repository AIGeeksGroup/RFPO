#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
CAN_CHECKPOINT="$CHECKPOINT_ROOT/95j3noe4_step_1000"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-can_source_prior_pilot}"

if [[ ! -f "$CAN_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Can checkpoint: $CAN_CHECKPOINT" >&2
  echo "Run research_scripts/remote/download_manip_checkpoints.sh first." >&2
  exit 1
fi

for prior_mode in gaussian previous_action; do
  for zero_sampling in True False; do
    output_dir="$RUNTIME_ROOT/results/${EXPERIMENT_TAG}_${prior_mode}_${zero_sampling}_$(timestamp)"
    python eval_checkpoint.py \
      --local-checkpoint-path "$CAN_CHECKPOINT" \
      --load-ema True \
      --eval-env Can \
      --eval-num-episodes "${EVAL_EPISODES:-20}" \
      --eval-num-envs "${EVAL_ENVS:-10}" \
      --seed "${EVAL_SEED:-20260808}" \
      --zero-sampling "$zero_sampling" \
      --source-prior-mode "$prior_mode" \
      --source-prior-sigma "${SOURCE_PRIOR_SIGMA:-0.5}" \
      --save-video False \
      --wandb-enable False \
      --output-dir "$output_dir" \
      2>&1 | tee "$RUNTIME_ROOT/logs/$(basename "$output_dir").log"
  done
done
