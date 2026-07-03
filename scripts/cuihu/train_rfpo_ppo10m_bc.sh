#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

RUN_NAME="${RUN_NAME:-rfpo_mustard_ppo10m_zero_chunk4_learn80_3m}"

nvidia-smi || true
bash scripts/server/train_state_single.sh "${SEQ}" "${RUN_NAME}" \
  "agent=fpo" \
  "agent.params.trust_region_mode=ppo" \
  "agent.params.bc_checkpoint=${BC_CKPT}" \
  "agent.params.bc_anchor_dataset=${BC_DATASET}" \
  "agent.params.bc_anchor_coef=0.02" \
  "agent.params.bc_anchor_min_coef=0.0" \
  "agent.params.bc_anchor_decay_steps=500000" \
  "agent.params.bc_anchor_batch_size=256" \
  "agent.params.action_anchor_coef=2.0" \
  "agent.params.action_anchor_min_coef=0.0" \
  "agent.params.action_anchor_decay_steps=500000" \
  "agent.params.action_anchor_batch_size=512" \
  "agent.params.action_anchor_loss=huber" \
  "agent.params.action_anchor_huber_delta=0.02" \
  "agent.params.on_policy_action_anchor_coef=0.2" \
  "agent.params.on_policy_action_anchor_min_coef=0.0" \
  "agent.params.on_policy_action_anchor_decay_steps=500000" \
  "agent.params.advantage_weighted_action_coef=0.05" \
  "agent.params.advantage_weighted_action_temp=0.7" \
  "agent.params.advantage_weighted_action_max_weight=10.0" \
  "agent.params.advantage_weighted_action_positive_only=True" \
  "agent.params.positive_advantage_only=True" \
  "agent.params.train_actor_after_iterations=1" \
  "agent.params.fpo_chunk_steps=4" \
  "agent.params.average_cfm_loss_in_chunk=True" \
  "agent.params.rollout_deterministic=True" \
  "agent.params.learning_rate=1.0e-5" \
  "agent.params.max_grad_norm=5.0" \
  "agent.params.actor_max_grad_norm=2.0" \
  "agent.params.critic_max_grad_norm=5.0" \
  "agent.params.clip_range=0.02" \
  "agent.params.cfm_loss_huber_delta=0.5" \
  "agent.params.cfm_loss_huber_style=fpo_control" \
  "agent.params.cfm_loss_clamp=4.0" \
  "agent.params.cfm_diff_clip=5.0" \
  "agent.params.n_steps=4096" \
  "agent.params.batch_size=256" \
  "agent.params.n_epochs=1" \
  "agent.params.sampling_steps=10" \
  "agent.params.n_samples_per_action=8" \
  "agent.params.best_metric=eval/mean_reward" \
  "agent.params.best_metric_mode=max" \
  "agent.params.save_best_model=True" \
  "agent.params.rollback_to_best_on_degrade=True" \
  "agent.params.rollback_patience=1" \
  "agent.params.reset_optimizer_on_rollback=True" \
  "agent.params.early_stop_on_degrade=False" \
  "agent.params.early_stop_patience=0" \
  "agent.params.degrade_metric=eval/mean_reward" \
  "agent.params.degrade_threshold=2.0" \
  "n_envs=8" \
  "n_eval_envs=2" \
  "total_timesteps=3000000" \
  "eval_freq=20000" \
  "save_freq=50000" \
  "restore_checkpoint_freq=50000" \
  "$@"
