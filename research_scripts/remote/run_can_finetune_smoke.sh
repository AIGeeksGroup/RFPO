#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT/manipulation_experiments"
source source_env.sh

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/manipulation_experiments/downloaded_checkpoints}"
CAN_CHECKPOINT="${CAN_CHECKPOINT:-$CHECKPOINT_ROOT/95j3noe4_step_1000}"
NUM_ENVS="${NUM_ENVS:-30}"
DATA_COLLECTION_STEPS="${DATA_COLLECTION_STEPS:-1600}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-$((NUM_ENVS * DATA_COLLECTION_STEPS))}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
N_ACTION_SAMPLES="${N_ACTION_SAMPLES:-8}"
ROLLOUT_FREQ="${ROLLOUT_FREQ:-1}"
RESET_CFM_INVALID_MASK_EACH_ITERATION="${RESET_CFM_INVALID_MASK_EACH_ITERATION:-False}"
ROLLOUT_ZERO_FRACTION="${ROLLOUT_ZERO_FRACTION:-0.0}"
ROLLOUT_TEMPERED_FRACTION="${ROLLOUT_TEMPERED_FRACTION:-0.0}"
ROLLOUT_TEMPERED_SCALE="${ROLLOUT_TEMPERED_SCALE:-0.5}"
GAE_LAMBDA="${GAE_LAMBDA:-0.99}"
ADVANTAGE_WEIGHTING="${ADVANTAGE_WEIGHTING:-signed}"
ADVANTAGE_WEIGHT_ESS_FRACTION="${ADVANTAGE_WEIGHT_ESS_FRACTION:-0.5}"
ADVANTAGE_NORMALIZATION_SCOPE="${ADVANTAGE_NORMALIZATION_SCOPE:-minibatch}"
SUCCESS_REPLAY_AUDIT="${SUCCESS_REPLAY_AUDIT:-False}"
SUCCESS_REPLAY_AUDIT_ITERATION="${SUCCESS_REPLAY_AUDIT_ITERATION:-2}"
SUCCESS_REPLAY_AUDIT_CHUNKS="${SUCCESS_REPLAY_AUDIT_CHUNKS:-64}"
SUCCESS_REPLAY_AUDIT_MIN_CHUNKS="${SUCCESS_REPLAY_AUDIT_MIN_CHUNKS:-32}"
MASTER_PORT="${MASTER_PORT:-29500}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-can_fpopp_smoke_seed${SEED}_$(timestamp)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNTIME_ROOT/results/$RUN_NAME}"

if [[ ! -f "$CAN_CHECKPOINT/policy/model.safetensors" ]]; then
  echo "Missing Can checkpoint: $CAN_CHECKPOINT" >&2
  exit 1
fi

torchrun --nproc_per_node=1 --master_port "$MASTER_PORT" finetune_online_rl.py \
  --distributed True \
  --base-policy-local-path "$CAN_CHECKPOINT" \
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
  --task Can \
  --eval-env Can \
  --eval-num-episodes "$EVAL_EPISODES" \
  --data-collection-steps "$DATA_COLLECTION_STEPS" \
  --do-chunk-level-ppo True \
  --eval-ema False \
  --exploration-noise-std None \
  --freeze-vision-encoder True \
  --gae-lambda "$GAE_LAMBDA" \
  --advantage-weighting "$ADVANTAGE_WEIGHTING" \
  --advantage-weight-ess-fraction "$ADVANTAGE_WEIGHT_ESS_FRACTION" \
  --advantage-normalization-scope "$ADVANTAGE_NORMALIZATION_SCOPE" \
  --success-replay-audit "$SUCCESS_REPLAY_AUDIT" \
  --success-replay-audit-iteration "$SUCCESS_REPLAY_AUDIT_ITERATION" \
  --success-replay-audit-chunks "$SUCCESS_REPLAY_AUDIT_CHUNKS" \
  --success-replay-audit-min-chunks "$SUCCESS_REPLAY_AUDIT_MIN_CHUNKS" \
  --n-action-samples "$N_ACTION_SAMPLES" \
  --n-action-steps 16 \
  --num-envs "$NUM_ENVS" \
  --sampling-steps 10 \
  --spo-clip-coef 0.01 \
  --zero-sampling True \
  --discount 0.99 \
  --sde-sigma 0 \
  --cfm-loss-average-group-size 1 \
  --reset-cfm-invalid-mask-each-iteration "$RESET_CFM_INVALID_MASK_EACH_ITERATION" \
  --rollout-zero-fraction "$ROLLOUT_ZERO_FRACTION" \
  --rollout-tempered-fraction "$ROLLOUT_TEMPERED_FRACTION" \
  --rollout-tempered-scale "$ROLLOUT_TEMPERED_SCALE" \
  --cfm-loss-use-huber True \
  --cfm-loss-huber-delta 0.5 \
  --clip-coef 0.02 \
  --max-grad-norm 5 \
  --clamp-logratio 5 \
  --clamp-old-cfm-loss 4 \
  --trust-region-mode ppo \
  --seed "$SEED"
