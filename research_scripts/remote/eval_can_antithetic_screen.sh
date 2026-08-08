#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
CAN_CHECKPOINT="${CAN_CHECKPOINT:-$CHECKPOINT_ROOT/95j3noe4_step_1000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_ENVS="${EVAL_ENVS:-16}"
SEED="${SEED:-20260824}"
RUN_PREFIX="${RUN_PREFIX:-can_step1000_antithetic_screen_seed${SEED}}"

if [[ ! -f "$CAN_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Can checkpoint: $CAN_CHECKPOINT" >&2
  exit 1
fi

for sampling_mode in zero antithetic_average; do
  output_dir="$RUNTIME_ROOT/results/${RUN_PREFIX}_${sampling_mode}"
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing output: $output_dir" >&2
    exit 1
  fi

  python eval_checkpoint.py \
    --local-checkpoint-path "$CAN_CHECKPOINT" \
    --load-ema True \
    --eval-env Can \
    --eval-num-episodes "$EVAL_EPISODES" \
    --eval-num-envs "$EVAL_ENVS" \
    --sampling-mode "$sampling_mode" \
    --sampling-steps 10 \
    --seed "$SEED" \
    --save-video False \
    --wandb-enable False \
    --output-dir "$output_dir" \
    2>&1 | tee "$RUNTIME_ROOT/logs/${RUN_PREFIX}_${sampling_mode}.log"
done
