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
LEARNING_RATE_ACTOR="${LEARNING_RATE_ACTOR:-1e-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
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
CFM_RATIO_GENERALIZATION_AUDIT="${CFM_RATIO_GENERALIZATION_AUDIT:-False}"
CFM_RATIO_GENERALIZATION_AUDIT_ITERATION="${CFM_RATIO_GENERALIZATION_AUDIT_ITERATION:-2}"
CFM_RATIO_GENERALIZATION_AUDIT_CHUNKS="${CFM_RATIO_GENERALIZATION_AUDIT_CHUNKS:-64}"
CFM_RATIO_GENERALIZATION_AUDIT_MIN_CHUNKS="${CFM_RATIO_GENERALIZATION_AUDIT_MIN_CHUNKS:-32}"
ADVANTAGE_STRATIFIED_MC_AUDIT="${ADVANTAGE_STRATIFIED_MC_AUDIT:-False}"
ADVANTAGE_STRATIFIED_MC_AUDIT_ITERATION="${ADVANTAGE_STRATIFIED_MC_AUDIT_ITERATION:-2}"
HELDOUT_RATIO_EARLY_STOP_AUDIT="${HELDOUT_RATIO_EARLY_STOP_AUDIT:-False}"
HELDOUT_RATIO_EARLY_STOP_AUDIT_ITERATION="${HELDOUT_RATIO_EARLY_STOP_AUDIT_ITERATION:-2}"
HELDOUT_RATIO_EARLY_STOP_AUDIT_CHUNKS="${HELDOUT_RATIO_EARLY_STOP_AUDIT_CHUNKS:-64}"
HELDOUT_RATIO_EARLY_STOP_THRESHOLD="${HELDOUT_RATIO_EARLY_STOP_THRESHOLD:-0.8}"
DISCOUNTED_SUCCESS_CRITIC_AUDIT="${DISCOUNTED_SUCCESS_CRITIC_AUDIT:-False}"
DISCOUNTED_SUCCESS_CRITIC_AUDIT_ITERATION="${DISCOUNTED_SUCCESS_CRITIC_AUDIT_ITERATION:-2}"
DISCOUNTED_SUCCESS_CRITIC_AUDIT_CHUNKS="${DISCOUNTED_SUCCESS_CRITIC_AUDIT_CHUNKS:-64}"
CRITIC_WARMUP_SCHEDULER_AUDIT="${CRITIC_WARMUP_SCHEDULER_AUDIT:-False}"
CRITIC_WARMUP_SCHEDULER_AUDIT_ITERATION="${CRITIC_WARMUP_SCHEDULER_AUDIT_ITERATION:-2}"
CRITIC_WARMUP_SCHEDULER_AUDIT_CHUNKS="${CRITIC_WARMUP_SCHEDULER_AUDIT_CHUNKS:-64}"
DIRECT_ADVANTAGE_AUDIT="${DIRECT_ADVANTAGE_AUDIT:-False}"
DIRECT_ADVANTAGE_AUDIT_ITERATION="${DIRECT_ADVANTAGE_AUDIT_ITERATION:-2}"
DIRECT_ADVANTAGE_AUDIT_CHUNKS="${DIRECT_ADVANTAGE_AUDIT_CHUNKS:-64}"
DIRECT_ADVANTAGE_CENTER_SAMPLES="${DIRECT_ADVANTAGE_CENTER_SAMPLES:-4}"
DIRECT_ADVANTAGE_HORIZON_CHUNKS="${DIRECT_ADVANTAGE_HORIZON_CHUNKS:-4}"
DIRECT_ADVANTAGE_CENTER_BATCH_SIZE="${DIRECT_ADVANTAGE_CENTER_BATCH_SIZE:-16}"
RANK_ADVANTAGE_AUDIT="${RANK_ADVANTAGE_AUDIT:-False}"
RANK_ADVANTAGE_AUDIT_ITERATION="${RANK_ADVANTAGE_AUDIT_ITERATION:-2}"
RANK_ADVANTAGE_AUDIT_CHUNKS="${RANK_ADVANTAGE_AUDIT_CHUNKS:-64}"
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
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
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
  --cfm-ratio-generalization-audit "$CFM_RATIO_GENERALIZATION_AUDIT" \
  --cfm-ratio-generalization-audit-iteration "$CFM_RATIO_GENERALIZATION_AUDIT_ITERATION" \
  --cfm-ratio-generalization-audit-chunks "$CFM_RATIO_GENERALIZATION_AUDIT_CHUNKS" \
  --cfm-ratio-generalization-audit-min-chunks "$CFM_RATIO_GENERALIZATION_AUDIT_MIN_CHUNKS" \
  --advantage-stratified-mc-audit "$ADVANTAGE_STRATIFIED_MC_AUDIT" \
  --advantage-stratified-mc-audit-iteration "$ADVANTAGE_STRATIFIED_MC_AUDIT_ITERATION" \
  --heldout-ratio-early-stop-audit "$HELDOUT_RATIO_EARLY_STOP_AUDIT" \
  --heldout-ratio-early-stop-audit-iteration "$HELDOUT_RATIO_EARLY_STOP_AUDIT_ITERATION" \
  --heldout-ratio-early-stop-audit-chunks "$HELDOUT_RATIO_EARLY_STOP_AUDIT_CHUNKS" \
  --heldout-ratio-early-stop-threshold "$HELDOUT_RATIO_EARLY_STOP_THRESHOLD" \
  --discounted-success-critic-audit "$DISCOUNTED_SUCCESS_CRITIC_AUDIT" \
  --discounted-success-critic-audit-iteration "$DISCOUNTED_SUCCESS_CRITIC_AUDIT_ITERATION" \
  --discounted-success-critic-audit-chunks "$DISCOUNTED_SUCCESS_CRITIC_AUDIT_CHUNKS" \
  --critic-warmup-scheduler-audit "$CRITIC_WARMUP_SCHEDULER_AUDIT" \
  --critic-warmup-scheduler-audit-iteration "$CRITIC_WARMUP_SCHEDULER_AUDIT_ITERATION" \
  --critic-warmup-scheduler-audit-chunks "$CRITIC_WARMUP_SCHEDULER_AUDIT_CHUNKS" \
  --direct-advantage-audit "$DIRECT_ADVANTAGE_AUDIT" \
  --direct-advantage-audit-iteration "$DIRECT_ADVANTAGE_AUDIT_ITERATION" \
  --direct-advantage-audit-chunks "$DIRECT_ADVANTAGE_AUDIT_CHUNKS" \
  --direct-advantage-center-samples "$DIRECT_ADVANTAGE_CENTER_SAMPLES" \
  --direct-advantage-horizon-chunks "$DIRECT_ADVANTAGE_HORIZON_CHUNKS" \
  --direct-advantage-center-batch-size "$DIRECT_ADVANTAGE_CENTER_BATCH_SIZE" \
  --rank-advantage-audit "$RANK_ADVANTAGE_AUDIT" \
  --rank-advantage-audit-iteration "$RANK_ADVANTAGE_AUDIT_ITERATION" \
  --rank-advantage-audit-chunks "$RANK_ADVANTAGE_AUDIT_CHUNKS" \
  --n-action-samples "$N_ACTION_SAMPLES" \
  --learning-rate-actor "$LEARNING_RATE_ACTOR" \
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
