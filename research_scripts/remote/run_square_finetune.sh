#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

OSMESA_ROOT="${FPO_OSMESA_ROOT:-$HOME/workspace/scratch/fpo-osmesa/osmesa-root}"
OSMESA_LIB="$OSMESA_ROOT/usr/lib/x86_64-linux-gnu"
if [[ ! -f "$OSMESA_LIB/libOSMesa.so.8" ]]; then
  echo "Missing OSMesa runtime: $OSMESA_LIB/libOSMesa.so.8" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$OSMESA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
SQUARE_CHECKPOINT="${SQUARE_CHECKPOINT:-$CHECKPOINT_ROOT/trc7rbt0_step_110000}"
NUM_ENVS="${NUM_ENVS:-16}"
DATA_COLLECTION_STEPS="${DATA_COLLECTION_STEPS:-320}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-25600}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
ROLLOUT_FREQ="${ROLLOUT_FREQ:-None}"
ZERO_ENDPOINT_PCGRAD_TRAIN="${ZERO_ENDPOINT_PCGRAD_TRAIN:-False}"
SEED="${SEED:-20260913}"
MASTER_PORT="${MASTER_PORT:-29500}"
RUN_NAME="${RUN_NAME:-square_fpopp_seed${SEED}_$(timestamp)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNTIME_ROOT/results/$RUN_NAME}"
LOG_PATH="${LOG_PATH:-$RUNTIME_ROOT/logs/$RUN_NAME.log}"

if [[ ! -f "$SQUARE_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Square checkpoint: $SQUARE_CHECKPOINT" >&2
  exit 1
fi

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

torchrun --nproc_per_node=1 --master_port "$MASTER_PORT" finetune_online_rl.py \
  --distributed True \
  --base-policy-local-path "$SQUARE_CHECKPOINT" \
  --load-ema True \
  --wandb-enable False \
  --experiment "$RUN_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --gradient-accumulation-steps 1 \
  --num-minibatches 8 \
  --log-freq 1 \
  --save-freq 1 \
  --rollout-freq "$ROLLOUT_FREQ" \
  --task Square \
  --eval-env Square \
  --eval-num-episodes "$EVAL_EPISODES" \
  --data-collection-steps "$DATA_COLLECTION_STEPS" \
  --do-chunk-level-ppo True \
  --eval-ema False \
  --exploration-noise-std None \
  --freeze-vision-encoder True \
  --gae-lambda 0.99 \
  --n-action-samples 8 \
  --learning-rate-actor 1e-5 \
  --n-action-steps 16 \
  --num-envs "$NUM_ENVS" \
  --sampling-steps 10 \
  --spo-clip-coef 0.01 \
  --zero-sampling True \
  --discount 0.995 \
  --sde-sigma 0 \
  --cfm-loss-average-group-size 1 \
  --cfm-loss-use-huber True \
  --cfm-loss-huber-delta 1 \
  --clip-coef 0.01 \
  --max-grad-norm 25 \
  --clamp-logratio None \
  --clamp-old-cfm-loss None \
  --trust-region-mode ppo \
  --zero-endpoint-pcgrad-train "$ZERO_ENDPOINT_PCGRAD_TRAIN" \
  --seed "$SEED" \
  2>&1 | tee "$LOG_PATH"
