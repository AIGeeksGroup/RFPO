#!/usr/bin/env bash
set -euo pipefail

ROOT="${CUIHU_ROOT:-/share/project/liyanjun/rongshanyu}"
REPO="${ROOT}/vividex_sapien_server_upload_fpo_control_full"
LOG_ROOT="${ROOT}/logs"
mkdir -p "${LOG_ROOT}"

cd "${REPO}"
source "${REPO}/scripts/cuihu/common.sh"

export VIVIDEX_HEADLESS_NO_RENDER=1
export HYDRA_FULL_ERROR=0
export SEQ="${SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"
export WANDB_GROUP="${WANDB_GROUP:-rfpo_scratch}"
export WANDB_PREFIX="${WANDB_PREFIX:-rfpo_scratch}"

launch_rfpo_scratch() {
  local gpu="$1"
  local run="$2"
  local lr="$3"
  local clip="$4"
  local mode="$5"
  local log_ratio="$6"
  local cfm_clip="$7"
  local residual_scale="$8"
  local noise="$9"
  local epochs="${10}"

  echo "[rfpo-scratch] launch ${run} on GPU${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash scripts/server/train_state_single.sh "${SEQ}" "${run}" \
      "agent=fpo" \
      "agent.params.actor_objective=fpo" \
      "agent.params.trust_region_mode=${mode}" \
      "agent.params.learning_rate=${lr}" \
      "agent.params.clip_range=${clip}" \
      "agent.params.log_ratio_scale=${log_ratio}" \
      "agent.params.cfm_diff_clip=${cfm_clip}" \
      "agent.params.cfm_loss_clamp=20.0" \
      "agent.params.action_head_mode=flow_residual" \
      "agent.params.residual_action_scale=${residual_scale}" \
      "agent.params.rollout_action_noise_std=${noise}" \
      "agent.params.rollout_deterministic=False" \
      "agent.params.eval_deterministic=False" \
      "agent.params.n_steps=4096" \
      "agent.params.batch_size=512" \
      "agent.params.n_epochs=${epochs}" \
      "agent.params.n_samples_per_action=16" \
      "agent.params.sampling_steps=8" \
      "agent.params.actor_hidden_dims=[512,512]" \
      "agent.params.critic_hidden_dims=[512,512]" \
      "agent.params.activation=elu" \
      "agent.params.max_grad_norm=1.0" \
      "agent.params.actor_max_grad_norm=0.5" \
      "agent.params.critic_max_grad_norm=2.0" \
      "agent.params.rollback_to_best_on_degrade=True" \
      "agent.params.rollback_patience=4" \
      "agent.params.degrade_threshold=3.0" \
      "n_envs=16" \
      "n_eval_envs=4" \
      "eval_n_episodes=25" \
      "total_timesteps=80000000" \
      "eval_freq=200000" \
      "save_freq=1000000" \
      "restore_checkpoint_freq=1000000" \
      "wandb.group=rfpo_scratch" \
      "wandb.sweep_name_prefix=${run}"
  ) > "${LOG_ROOT}/${run}.out" 2>&1 &
  echo "$!" > "${LOG_ROOT}/${run}.pid"
}

launch_rfpo_scratch 0 rfpo_scratch_fpo_default_80m 1.0e-5 0.20 ppo 1.0 5.0 0.10 0.00 5
launch_rfpo_scratch 1 rfpo_scratch_aspo_conservative_80m 5.0e-6 0.10 aspo 0.5 2.0 0.05 0.02 3
launch_rfpo_scratch 2 rfpo_scratch_spo_explore_80m 8.0e-6 0.05 spo 0.5 2.0 0.10 0.04 4

echo "[rfpo-scratch] launched jobs:"
jobs -l
wait
