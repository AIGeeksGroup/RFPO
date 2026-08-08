#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
CAN_CHECKPOINT="$CHECKPOINT_ROOT/95j3noe4_step_1000"

if [[ ! -f "$CAN_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Can checkpoint: $CAN_CHECKPOINT" >&2
  echo "Run research_scripts/remote/download_manip_checkpoints.sh first." >&2
  exit 1
fi

for zero_sampling in True False; do
  python eval_checkpoint.py \
    --local_checkpoint_path "$CAN_CHECKPOINT" \
    --load-ema True \
    --eval_env Can \
    --eval_num_episodes "${EVAL_EPISODES:-200}" \
    --eval-num-envs "${EVAL_ENVS:-50}" \
    --zero-sampling "$zero_sampling" \
    --save-video False \
    --wandb-enable False \
    --output-dir "$RUNTIME_ROOT/results/can_base_${zero_sampling}_$(timestamp)"
done
