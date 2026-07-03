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
export WANDB_GROUP="${WANDB_GROUP:-rm75_rfpo_scratch}"
export WANDB_PREFIX="${WANDB_PREFIX:-rm75_rfpo}"
export VIVIDEX_RM75_URDF_OVERRIDE="${REPO}/assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_driven12.urdf"

RM75_TRAJ="norm_trajectories/rm75_inspire_right/${SEQ}.npz"
if [[ ! -f "${RM75_TRAJ}" ]]; then
  python tools/retarget_rm75_inspire_reference.py \
    --src "norm_trajectories/${SEQ}.npz" \
    --dst "${RM75_TRAJ}" \
    --overwrite
else
  echo "[rm75-rfpo] using existing retargeted trajectory: ${RM75_TRAJ}"
fi

launch_rm75_rfpo() {
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

  echo "[rm75-rfpo] launch ${run} on GPU${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash scripts/server/train_state_single.sh "${SEQ}" "${run}" \
      "agent=fpo" \
      "env.robot_name=rm75_inspire_right" \
      "env.norm_traj=True" \
      "env.info_keywords=[pregrasp_success,pregrasp_steps,imitate_steps,hand_jpos_err,obj_com_err,obj_rot_err,obj_lift,contact_count,stable_grasp_contact,hand_mjpos_err,stage,control_error,obj_tgt_dist]" \
      "+env.task_kwargs.reward_kwargs.pregrasp_success_thresh=0.075" \
      "+env.task_kwargs.reward_kwargs.no_contact_grace_steps=30" \
      "+env.task_kwargs.reward_kwargs.obj_com_done_thresh=0.18" \
      "+env.task_kwargs.reward_kwargs.object_reward_requires_contact=True" \
      "+env.task_kwargs.reward_kwargs.no_contact_penalty=0.25" \
      "+env.task_kwargs.reward_kwargs.contact_reward_scale=0.8" \
      "+env.task_kwargs.reward_kwargs.stable_contact_bonus=1.2" \
      "+env.task_kwargs.reward_kwargs.required_non_thumb_contacts=1" \
      "env.task_kwargs.reward_kwargs.object_reward_scale=10.0" \
      "env.task_kwargs.reward_kwargs.obj_err_scale=50.0" \
      "+env.task_kwargs.reward_kwargs.hand_mimic_reward_scale=3.0" \
      "env.task_kwargs.reward_kwargs.lift_bonus_thresh=0.015" \
      "env.task_kwargs.reward_kwargs.lift_bonus_mag=3.0" \
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
      "agent.params.best_metric=eval/mean_reward" \
      "agent.params.save_best_model=True" \
      "agent.params.rollback_to_best_on_degrade=True" \
      "agent.params.rollback_patience=4" \
      "agent.params.degrade_threshold=2.0" \
      "n_envs=16" \
      "n_eval_envs=4" \
      "eval_n_episodes=25" \
      "total_timesteps=800000" \
      "eval_freq=50000" \
      "save_freq=100000" \
      "restore_checkpoint_freq=100000" \
      "wandb.group=rm75_rfpo_scratch" \
      "wandb.sweep_name_prefix=${run}"
  ) > "${LOG_ROOT}/${run}.out" 2>&1 &
  echo "$!" > "${LOG_ROOT}/${run}.pid"
}

launch_rm75_rfpo 0 rm75_rfpo_scratch_fpo_800k 1.0e-5 0.20 ppo 1.0 5.0 0.10 0.02 5
launch_rm75_rfpo 1 rm75_rfpo_scratch_aspo_800k 6.0e-6 0.12 aspo 0.5 2.0 0.07 0.03 4
launch_rm75_rfpo 2 rm75_rfpo_scratch_spo_800k 8.0e-6 0.08 spo 0.5 2.0 0.10 0.04 4

echo "[rm75-rfpo] launched jobs:"
jobs -l
wait
