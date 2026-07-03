#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

SEQ="${SEQ:?SEQ is required}"
OBJECT_TAG="${OBJECT_TAG:?OBJECT_TAG is required}"
RUN_PREFIX="${RUN_PREFIX:-rfpo16_${OBJECT_TAG}}"
RFPO_FINAL_TARGET="${RFPO_FINAL_TARGET:-20000000}"
EVAL_FREQ="${EVAL_FREQ:-200000}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-25}"
SAVE_FREQ="${SAVE_FREQ:-1000000}"
RESTORE_FREQ="${RESTORE_FREQ:-1000000}"

export VIVIDEX_N_ENVS="${VIVIDEX_N_ENVS:-16}"
export VIVIDEX_N_EVAL_ENVS="${VIVIDEX_N_EVAL_ENVS:-4}"
export WANDB_GROUP="${WANDB_GROUP:-rfpo_scratch_16block}"

BLOCK_NAMES=(
  warmup_std20_to12m
  best70_low_lr3_to13m
  best71_ultralow_to134m
  best72_micro_to138m
  best73_nano_to142m
  best73_pico_to146m
  best74_femto_to19m
  best75_hybridmicro_to196m
  best75_guardppo_to202m
  best75_elite_nll_b_to21m
  best75_elite_joint_b_to22m
  best75_elite_joint_c_to242m
  best75_precision_residual_b_to258m
  best75_precision_residual_b_to284m
  best75_elite_nll_clean_to294m
  best76_prec_lr25_to302m
)

# These are cumulative targets, matching the historical RFPO-from-scratch
# refinement chain. The first block deliberately runs through 12M so the
# critical 0-5M and 5-10M warmup behavior is produced inside one uninterrupted
# high-noise run.
BLOCK_TARGETS=(
  12000000
  13000000
  13400000
  13800000
  14200000
  14600000
  19000000
  19600000
  20200000
  21000000
  22000000
  24200000
  25800000
  28400000
  29400000
  30200000
)

BLOCK_OVERRIDES=(
  "agent.params.actor_objective=gaussian_ppo agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=1e-5 agent.params.min_learning_rate=1e-5 agent.params.max_learning_rate=1e-5 agent.params.gaussian_action_std=0.20 agent.params.gaussian_action_std_decay_steps=5000000 agent.params.gaussian_action_std_min_scale=0.25 agent.params.clip_range=0.20 agent.params.residual_action_scale=0.10 agent.params.on_policy_action_anchor_coef=0.0 agent.params.n_epochs=5 agent.params.max_grad_norm=5.0 agent.params.actor_max_grad_norm=2.0 agent.params.critic_max_grad_norm=5.0 agent.params.best_metric=eval/mean_reward agent.params.rollback_to_best_on_degrade=false"
  "agent.params.actor_objective=gaussian_ppo agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=3e-6 agent.params.min_learning_rate=3e-6 agent.params.max_learning_rate=3e-6 agent.params.gaussian_action_std=0.050 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.100 agent.params.residual_action_scale=0.050 agent.params.on_policy_action_anchor_coef=0.0 agent.params.n_epochs=4 agent.params.max_grad_norm=5.0 agent.params.actor_max_grad_norm=2.0 agent.params.critic_max_grad_norm=5.0 agent.params.best_metric=eval/mean_reward agent.params.rollback_to_best_on_degrade=false"
  "agent.params.actor_objective=gaussian_ppo agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=1e-6 agent.params.min_learning_rate=1e-6 agent.params.max_learning_rate=1e-6 agent.params.gaussian_action_std=0.045 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.060 agent.params.residual_action_scale=0.035 agent.params.on_policy_action_anchor_coef=0.0 agent.params.n_epochs=3 agent.params.max_grad_norm=5.0 agent.params.actor_max_grad_norm=1.0 agent.params.critic_max_grad_norm=5.0 agent.params.best_metric=eval/mean_reward agent.params.rollback_to_best_on_degrade=false"
  "agent.params.actor_objective=gaussian_ppo agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=5e-7 agent.params.min_learning_rate=5e-7 agent.params.max_learning_rate=5e-7 agent.params.gaussian_action_std=0.035 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.040 agent.params.residual_action_scale=0.025 agent.params.on_policy_action_anchor_coef=0.0 agent.params.n_epochs=2 agent.params.max_grad_norm=3.0 agent.params.actor_max_grad_norm=0.5 agent.params.critic_max_grad_norm=3.0 agent.params.best_metric=eval/mean_reward agent.params.rollback_to_best_on_degrade=false"
  "agent.params.actor_objective=gaussian_ppo agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=3e-7 agent.params.min_learning_rate=3e-7 agent.params.max_learning_rate=3e-7 agent.params.gaussian_action_std=0.030 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.030 agent.params.residual_action_scale=0.020 agent.params.on_policy_action_anchor_coef=0.0 agent.params.n_epochs=2 agent.params.max_grad_norm=2.0 agent.params.actor_max_grad_norm=0.35 agent.params.critic_max_grad_norm=2.0 agent.params.best_metric=eval/mean_reward agent.params.rollback_to_best_on_degrade=false"
  "agent.params.actor_objective=gaussian_ppo agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=2e-7 agent.params.min_learning_rate=2e-7 agent.params.max_learning_rate=2e-7 agent.params.gaussian_action_std=0.026 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.025 agent.params.residual_action_scale=0.018 agent.params.on_policy_action_anchor_coef=0.0 agent.params.n_epochs=2 agent.params.max_grad_norm=1.5 agent.params.actor_max_grad_norm=0.25 agent.params.critic_max_grad_norm=1.5 agent.params.best_metric=eval/mean_reward agent.params.rollback_to_best_on_degrade=false"
  "agent.params.actor_objective=gaussian_ppo agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=1e-7 agent.params.min_learning_rate=1e-7 agent.params.max_learning_rate=1e-7 agent.params.gaussian_action_std=0.022 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.020 agent.params.residual_action_scale=0.015 agent.params.on_policy_action_anchor_coef=0.0 agent.params.n_epochs=2 agent.params.max_grad_norm=1.0 agent.params.actor_max_grad_norm=0.2 agent.params.critic_max_grad_norm=1.0 agent.params.best_metric=eval/mean_reward agent.params.rollback_to_best_on_degrade=false"
  "agent.params.actor_objective=hybrid_fpo agent.params.fpo_objective_coef=0.5 agent.params.gaussian_objective_coef=1.0 agent.params.trust_region_mode=aspo agent.params.action_head_mode=flow_residual agent.params.learning_rate=6e-8 agent.params.min_learning_rate=6e-8 agent.params.max_learning_rate=6e-8 agent.params.gaussian_action_std=0.018 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.015 agent.params.residual_action_scale=0.012 agent.params.on_policy_action_anchor_coef=0.03 agent.params.on_policy_action_anchor_min_coef=0.03 agent.params.n_epochs=2 agent.params.max_grad_norm=0.8 agent.params.actor_max_grad_norm=0.08 agent.params.critic_max_grad_norm=0.8 agent.params.best_metric=eval/mean_reward agent.params.rollback_to_best_on_degrade=false agent.params.target_kl=0.001 agent.params.max_clip_fraction=0.18"
  "agent.params.actor_objective=gaussian_ppo agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=2e-7 agent.params.min_learning_rate=2e-7 agent.params.max_learning_rate=2e-7 agent.params.gaussian_action_std=0.022 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.025 agent.params.residual_action_scale=0.014 agent.params.on_policy_action_anchor_coef=0.10 agent.params.on_policy_action_anchor_min_coef=0.10 agent.params.n_epochs=1 agent.params.max_grad_norm=0.6 agent.params.actor_max_grad_norm=0.1 agent.params.critic_max_grad_norm=0.6 agent.params.best_metric=eval/mean_reward agent.params.rollback_to_best_on_degrade=false agent.params.target_kl=0.0008 agent.params.max_clip_fraction=0.12"
  "agent.params.actor_objective=hybrid_fpo agent.params.fpo_objective_coef=0.35 agent.params.gaussian_objective_coef=1.0 agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=8e-7 agent.params.min_learning_rate=8e-7 agent.params.max_learning_rate=8e-7 agent.params.gaussian_action_std=0.016 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.035 agent.params.residual_action_scale=0.014 agent.params.on_policy_action_anchor_coef=0.065 agent.params.on_policy_action_anchor_min_coef=0.065 agent.params.precision_elite_action_coef=0.025 agent.params.precision_elite_mode=gaussian_nll agent.params.precision_elite_err_threshold=0.003 agent.params.precision_elite_top_fraction=0.08 agent.params.precision_elite_min_fraction=0.03 agent.params.precision_elite_reward_quantile=0.78 agent.params.n_epochs=1 agent.params.max_grad_norm=0.7 agent.params.actor_max_grad_norm=0.1 agent.params.critic_max_grad_norm=0.7 agent.params.best_metric=eval/mean_reward agent.params.rollback_to_best_on_degrade=false agent.params.target_kl=0.0015 agent.params.max_clip_fraction=0.18"
  "agent.params.actor_objective=hybrid_fpo agent.params.fpo_objective_coef=0.35 agent.params.gaussian_objective_coef=1.0 agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=8e-7 agent.params.min_learning_rate=8e-7 agent.params.max_learning_rate=8e-7 agent.params.gaussian_action_std=0.016 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.035 agent.params.residual_action_scale=0.014 agent.params.on_policy_action_anchor_coef=0.065 agent.params.on_policy_action_anchor_min_coef=0.065 agent.params.precision_elite_action_coef=0.050 agent.params.precision_elite_mode=gaussian_nll agent.params.precision_elite_err_threshold=0.0025 agent.params.precision_elite_top_fraction=0.065 agent.params.precision_elite_min_fraction=0.03 agent.params.precision_elite_reward_quantile=0.82 agent.params.n_epochs=1 agent.params.max_grad_norm=0.7 agent.params.actor_max_grad_norm=0.1 agent.params.critic_max_grad_norm=0.7 agent.params.best_metric=eval/mean_reward agent.params.rollback_to_best_on_degrade=false agent.params.target_kl=0.0015 agent.params.max_clip_fraction=0.18"
  "agent.params.actor_objective=gaussian_ppo agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=1.5e-6 agent.params.min_learning_rate=1.5e-6 agent.params.max_learning_rate=1.5e-6 agent.params.gaussian_action_std=0.014 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.040 agent.params.residual_action_scale=0.014 agent.params.on_policy_action_anchor_coef=0.05 agent.params.on_policy_action_anchor_min_coef=0.05 agent.params.precision_elite_action_coef=0.036 agent.params.precision_elite_mode=gaussian_nll agent.params.precision_elite_err_threshold=0.0024 agent.params.precision_elite_top_fraction=0.06 agent.params.precision_elite_min_fraction=0.03 agent.params.precision_elite_reward_quantile=0.83 agent.params.n_epochs=1 agent.params.max_grad_norm=0.55 agent.params.actor_max_grad_norm=0.08 agent.params.critic_max_grad_norm=0.55 agent.params.best_metric=eval/rfpo_precision_score agent.params.rollback_to_best_on_degrade=false agent.params.target_kl=0.0012 agent.params.max_clip_fraction=0.15"
  "agent.params.actor_objective=hybrid_fpo agent.params.fpo_objective_coef=0.35 agent.params.gaussian_objective_coef=1.0 agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_plus_mlp_residual agent.params.learning_rate=8e-7 agent.params.min_learning_rate=8e-7 agent.params.max_learning_rate=8e-7 agent.params.gaussian_action_std=0.012 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.030 agent.params.residual_action_scale=0.006 agent.params.on_policy_action_anchor_coef=0.05 agent.params.on_policy_action_anchor_min_coef=0.05 agent.params.precision_elite_action_coef=0.120 agent.params.precision_elite_mode=action agent.params.precision_elite_err_threshold=0.0023 agent.params.precision_elite_top_fraction=0.055 agent.params.precision_elite_min_fraction=0.025 agent.params.precision_elite_reward_quantile=0.84 agent.params.n_epochs=1 agent.params.max_grad_norm=0.55 agent.params.actor_max_grad_norm=0.06 agent.params.critic_max_grad_norm=0.55 agent.params.best_metric=eval/rfpo_precision_score agent.params.rollback_to_best_on_degrade=false agent.params.target_kl=0.0012 agent.params.max_clip_fraction=0.15"
  "agent.params.actor_objective=hybrid_fpo agent.params.fpo_objective_coef=0.35 agent.params.gaussian_objective_coef=1.0 agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_plus_mlp_residual agent.params.learning_rate=8e-7 agent.params.min_learning_rate=8e-7 agent.params.max_learning_rate=8e-7 agent.params.gaussian_action_std=0.012 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.030 agent.params.residual_action_scale=0.006 agent.params.on_policy_action_anchor_coef=0.05 agent.params.on_policy_action_anchor_min_coef=0.05 agent.params.precision_elite_action_coef=0.120 agent.params.precision_elite_mode=action agent.params.precision_elite_err_threshold=0.0023 agent.params.precision_elite_top_fraction=0.055 agent.params.precision_elite_min_fraction=0.025 agent.params.precision_elite_reward_quantile=0.84 agent.params.n_epochs=1 agent.params.max_grad_norm=0.55 agent.params.actor_max_grad_norm=0.06 agent.params.critic_max_grad_norm=0.55 agent.params.best_metric=eval/rfpo_precision_score agent.params.rollback_to_best_on_degrade=false agent.params.target_kl=0.0012 agent.params.max_clip_fraction=0.15"
  "agent.params.actor_objective=gaussian_ppo agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=5e-7 agent.params.min_learning_rate=5e-7 agent.params.max_learning_rate=5e-7 agent.params.gaussian_action_std=0.014 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.025 agent.params.residual_action_scale=0.012 agent.params.on_policy_action_anchor_coef=0.05 agent.params.on_policy_action_anchor_min_coef=0.05 agent.params.precision_elite_action_coef=0.025 agent.params.precision_elite_mode=gaussian_nll agent.params.precision_elite_err_threshold=0.0022 agent.params.precision_elite_top_fraction=0.050 agent.params.precision_elite_min_fraction=0.025 agent.params.precision_elite_reward_quantile=0.85 agent.params.n_epochs=1 agent.params.max_grad_norm=0.55 agent.params.actor_max_grad_norm=0.08 agent.params.critic_max_grad_norm=0.55 agent.params.best_metric=eval/rfpo_precision_score agent.params.rollback_to_best_on_degrade=false agent.params.target_kl=0.0012 agent.params.max_clip_fraction=0.14"
  "agent.params.actor_objective=gaussian_ppo agent.params.trust_region_mode=ppo agent.params.action_head_mode=flow_residual agent.params.learning_rate=2.5e-7 agent.params.min_learning_rate=2.5e-7 agent.params.max_learning_rate=2.5e-7 agent.params.gaussian_action_std=0.008 agent.params.gaussian_action_std_decay_steps=0 agent.params.gaussian_action_std_min_scale=1.0 agent.params.clip_range=0.014 agent.params.residual_action_scale=0.004 agent.params.on_policy_action_anchor_coef=0.10 agent.params.on_policy_action_anchor_min_coef=0.10 agent.params.precision_elite_action_coef=0.050 agent.params.precision_elite_mode=gaussian_nll agent.params.precision_elite_err_threshold=0.0021 agent.params.precision_elite_top_fraction=0.045 agent.params.precision_elite_min_fraction=0.025 agent.params.precision_elite_reward_quantile=0.86 agent.params.n_epochs=1 agent.params.max_grad_norm=0.34 agent.params.actor_max_grad_norm=0.04 agent.params.critic_max_grad_norm=0.34 agent.params.best_metric=eval/rfpo_precision_score agent.params.rollback_to_best_on_degrade=false agent.params.target_kl=0.00045 agent.params.max_clip_fraction=0.06"
)

resume_model="null"
for idx in "${!BLOCK_NAMES[@]}"; do
  block="${BLOCK_NAMES[$idx]}"
  target="${BLOCK_TARGETS[$idx]}"
  if (( target > RFPO_FINAL_TARGET )); then
    target="${RFPO_FINAL_TARGET}"
  fi
  if (( target <= 0 )); then
    continue
  fi
  run_name="${RUN_PREFIX}_b$(printf '%02d' "$idx")_${block}"
  run_dir="${VIVIDEX_RUNTIME_ROOT}/results/state_baseline/${run_name}"
  export WANDB_PREFIX="${run_name}"

  resume_for_block="${resume_model}"
  if [[ "${resume_for_block}" == "null" ]]; then
    if [[ -f "${run_dir}/models/last.pt" ]]; then
      resume_for_block="${run_dir}/models/last.pt"
    elif [[ -f "${run_dir}/restore_checkpoint.pt" ]]; then
      resume_for_block="${run_dir}/restore_checkpoint.pt"
    fi
  fi

  echo "[rfpo16] $(date '+%F %T') object=${OBJECT_TAG} block=${idx}/${#BLOCK_NAMES[@]} run=${run_name} target=${target}/${RFPO_FINAL_TARGET}"
  echo "[rfpo16] resume_model=${resume_for_block}"

  read -r -a block_args <<< "${BLOCK_OVERRIDES[$idx]}"
  bash scripts/server/train_state_single.sh "${SEQ}" "${run_name}" \
    "agent=fpo" \
    "resume_model=${resume_for_block}" \
    "agent.params.resume_load_optimizer=false" \
    "agent.params.bc_checkpoint=null" \
    "agent.params.bc_anchor_dataset=null" \
    "agent.params.ppo_base_checkpoint=null" \
    "agent.params.ppo_teacher_checkpoint=null" \
    "agent.params.trainable_ppo_base_actor=false" \
    "agent.params.trainable_ppo_base_critic=false" \
    "n_envs=${VIVIDEX_N_ENVS}" \
    "n_eval_envs=${VIVIDEX_N_EVAL_ENVS}" \
    "total_timesteps=${target}" \
    "eval_freq=${EVAL_FREQ}" \
    "eval_n_episodes=${EVAL_N_EPISODES}" \
    "save_freq=${SAVE_FREQ}" \
    "restore_checkpoint_freq=${RESTORE_FREQ}" \
    "env.task_kwargs.reward_kwargs.obj_err_scale=50" \
    "env.task_kwargs.reward_kwargs.object_reward_scale=10.0" \
    "${block_args[@]}"

  next="${VIVIDEX_RUNTIME_ROOT}/results/state_baseline/${run_name}/models/last.pt"
  if [[ ! -f "${next}" ]]; then
    next="${VIVIDEX_RUNTIME_ROOT}/results/state_baseline/${run_name}/restore_checkpoint.pt"
    if [[ ! -f "${next}" ]]; then
      echo "[rfpo16] ERROR: expected final checkpoint missing: ${next}" >&2
      exit 1
    fi
  fi
  resume_model="${next}"
  if (( target >= RFPO_FINAL_TARGET )); then
    break
  fi
done

echo "[rfpo16] completed object=${OBJECT_TAG}; final_checkpoint=${resume_model}"
