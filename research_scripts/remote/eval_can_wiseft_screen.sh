#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

SOURCE_RUN="${SOURCE_RUN:-$RUNTIME_ROOT/results/can_step6000_advnorm_minibatch_5iter_seed20260812}"
ANCHOR_CHECKPOINT="${ANCHOR_CHECKPOINT:-$SOURCE_RUN/checkpoints/step_48000}"
FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-$SOURCE_RUN/checkpoints/step_240000}"
MIDPOINT_CHECKPOINT="${MIDPOINT_CHECKPOINT:-$RUNTIME_ROOT/results/can_step6000_wiseft_midpoint_seed20260812}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_ENVS="${EVAL_ENVS:-16}"
SEED="${SEED:-20260823}"
RUN_PREFIX="${RUN_PREFIX:-can_step6000_wiseft_screen_seed${SEED}}"

declare -a CONDITIONS=(
  "anchor:$ANCHOR_CHECKPOINT"
  "finetuned:$FINETUNED_CHECKPOINT"
  "midpoint:$MIDPOINT_CHECKPOINT"
)

for condition in "${CONDITIONS[@]}"; do
  label="${condition%%:*}"
  checkpoint="${condition#*:}"
  if [[ ! -f "$checkpoint/policy/model.safetensors" ]]; then
    echo "Missing checkpoint for $label: $checkpoint" >&2
    exit 1
  fi

  for mode in zero random; do
    if [[ "$mode" == "zero" ]]; then
      zero_sampling=True
    else
      zero_sampling=False
    fi
    output_dir="$RUNTIME_ROOT/results/${RUN_PREFIX}_${label}_${mode}"
    if [[ -e "$output_dir" ]]; then
      echo "Refusing to overwrite existing output: $output_dir" >&2
      exit 1
    fi

    python eval_checkpoint.py \
      --local-checkpoint-path "$checkpoint" \
      --load-ema False \
      --eval-env Can \
      --eval-num-episodes "$EVAL_EPISODES" \
      --eval-num-envs "$EVAL_ENVS" \
      --zero-sampling "$zero_sampling" \
      --sampling-steps 10 \
      --seed "$SEED" \
      --save-video False \
      --wandb-enable False \
      --output-dir "$output_dir"
  done
done
