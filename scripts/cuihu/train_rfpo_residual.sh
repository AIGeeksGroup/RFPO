#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

RUN_NAME="${RUN_NAME:-rfpo_residual_debug}"
GPU_ID="${GPU_ID:-0}"
CLEAN_RUN="${CLEAN_RUN:-0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1200000}"
ACTION_HEAD_MODE="${ACTION_HEAD_MODE:-mlp_residual}"
RESIDUAL_ACTION_SCALE="${RESIDUAL_ACTION_SCALE:-0.30}"
FLOW_RESIDUAL_COEF="${FLOW_RESIDUAL_COEF:-0.0}"
GAUSSIAN_ACTION_STD="${GAUSSIAN_ACTION_STD:-0.08}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
CLIP_RANGE="${CLIP_RANGE:-0.2}"
ENT_COEF="${ENT_COEF:-0.0}"
N_ENVS="${N_ENVS:-8}"
N_EVAL_ENVS="${N_EVAL_ENVS:-2}"
N_STEPS="${N_STEPS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-512}"
N_EPOCHS="${N_EPOCHS:-4}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"

VALUE_BC_CKPT="${VALUE_BC_CKPT:-${CUIHU_ROOT}/results/fpo_bc_mustard_ppo10m_value/flow_bc_value_distilled.pt}"
RFPO_BC_CKPT="${RFPO_BC_CKPT:-${VALUE_BC_CKPT}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VIVIDEX_HEADLESS_NO_RENDER=1
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-0}"

RUN_DIR="${VIVIDEX_RESULTS_ROOT}/state_baseline/${RUN_NAME}"
if [[ "${CLEAN_RUN}" == "1" ]]; then
  rm -rf "${RUN_DIR}"
fi

echo "[rfpo-residual] run=${RUN_NAME}"
echo "[rfpo-residual] gpu=${CUDA_VISIBLE_DEVICES}"
echo "[rfpo-residual] action_head_mode=${ACTION_HEAD_MODE}"
echo "[rfpo-residual] residual_action_scale=${RESIDUAL_ACTION_SCALE}"
echo "[rfpo-residual] flow_residual_coef=${FLOW_RESIDUAL_COEF}"
echo "[rfpo-residual] gaussian_action_std=${GAUSSIAN_ACTION_STD}"
echo "[rfpo-residual] bc_checkpoint=${RFPO_BC_CKPT}"
echo "[rfpo-residual] ppo_base=${PPO_10M}"
nvidia-smi || true

bash scripts/server/train_state_single.sh "${SEQ}" "${RUN_NAME}" \
  "agent=fpo" \
  "agent.params.trust_region_mode=ppo" \
  "agent.params.actor_objective=gaussian_ppo" \
  "agent.params.action_head_mode=${ACTION_HEAD_MODE}" \
  "agent.params.residual_action_scale=${RESIDUAL_ACTION_SCALE}" \
  "agent.params.flow_residual_coef=${FLOW_RESIDUAL_COEF}" \
  "agent.params.ppo_base_checkpoint=${PPO_10M}" \
  "agent.params.ppo_teacher_checkpoint=${PPO_10M}" \
  "agent.params.bc_checkpoint=${RFPO_BC_CKPT}" \
  "agent.params.bc_anchor_dataset=${BC_DATASET}" \
  "agent.params.gaussian_action_std=${GAUSSIAN_ACTION_STD}" \
  "agent.params.ent_coef=${ENT_COEF}" \
  "agent.params.learning_rate=${LEARNING_RATE}" \
  "agent.params.clip_range=${CLIP_RANGE}" \
  "agent.params.max_grad_norm=5.0" \
  "agent.params.actor_max_grad_norm=1.0" \
  "agent.params.critic_max_grad_norm=5.0" \
  "agent.params.bc_anchor_coef=0.001" \
  "agent.params.bc_anchor_min_coef=0.0" \
  "agent.params.bc_anchor_decay_steps=300000" \
  "agent.params.bc_anchor_batch_size=256" \
  "agent.params.action_anchor_coef=0.0" \
  "agent.params.on_policy_action_anchor_coef=0.0" \
  "agent.params.ppo_teacher_action_anchor_coef=0.01" \
  "agent.params.ppo_teacher_action_anchor_min_coef=0.0" \
  "agent.params.ppo_teacher_action_anchor_decay_steps=300000" \
  "agent.params.advantage_weighted_action_coef=0.02" \
  "agent.params.advantage_weighted_action_temp=0.7" \
  "agent.params.advantage_weighted_action_max_weight=10.0" \
  "agent.params.advantage_weighted_action_positive_only=True" \
  "agent.params.advantage_weighted_action_mode=action" \
  "agent.params.positive_advantage_only=False" \
  "agent.params.rollout_deterministic=False" \
  "agent.params.n_steps=${N_STEPS}" \
  "agent.params.batch_size=${BATCH_SIZE}" \
  "agent.params.n_epochs=${N_EPOCHS}" \
  "agent.params.sampling_steps=8" \
  "agent.params.n_samples_per_action=16" \
  "agent.params.best_metric=eval/mean_reward" \
  "agent.params.best_metric_mode=max" \
  "agent.params.save_best_model=True" \
  "agent.params.rollback_to_best_on_degrade=True" \
  "agent.params.rollback_patience=2" \
  "agent.params.reset_optimizer_on_rollback=True" \
  "agent.params.early_stop_on_degrade=False" \
  "agent.params.degrade_metric=eval/mean_reward" \
  "agent.params.degrade_threshold=4.0" \
  "n_envs=${N_ENVS}" \
  "n_eval_envs=${N_EVAL_ENVS}" \
  "total_timesteps=${TOTAL_TIMESTEPS}" \
  "eval_freq=${EVAL_FREQ}" \
  "save_freq=${SAVE_FREQ}" \
  "restore_checkpoint_freq=${SAVE_FREQ}" \
  "$@"
