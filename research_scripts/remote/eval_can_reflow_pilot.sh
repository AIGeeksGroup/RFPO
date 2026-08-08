#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
EVAL_TAG="${EVAL_TAG:-can_reflow_pilot}"

if [[ ! -f "$CHECKPOINT_PATH/policy/model.safetensors" ]]; then
  echo "Set CHECKPOINT_PATH to a saved control or reflow checkpoint." >&2
  exit 1
fi

for sampling_steps in 10 4; do
  for zero_sampling in True False; do
    output_dir="$RUNTIME_ROOT/results/${EVAL_TAG}_nfe${sampling_steps}_${zero_sampling}_$(timestamp)"
    python eval_checkpoint.py \
      --local-checkpoint-path "$CHECKPOINT_PATH" \
      --load-ema True \
      --eval-env Can \
      --eval-num-episodes "${EVAL_EPISODES:-20}" \
      --eval-num-envs "${EVAL_ENVS:-10}" \
      --seed "${EVAL_SEED:-20260808}" \
      --zero-sampling "$zero_sampling" \
      --source-prior-mode gaussian \
      --sampling-steps "$sampling_steps" \
      --save-video False \
      --wandb-enable False \
      --output-dir "$output_dir" \
      2>&1 | tee "$RUNTIME_ROOT/logs/$(basename "$output_dir").log"
  done
done
