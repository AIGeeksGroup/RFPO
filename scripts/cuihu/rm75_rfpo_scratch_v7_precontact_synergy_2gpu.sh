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
export RM75_RUN_TAG="${RM75_RUN_TAG:-v7_precontact_synergy}"
export WANDB_GROUP="${WANDB_GROUP:-rm75_rfpo_scratch_2gpu}"
export WANDB_PREFIX="${WANDB_PREFIX:-rm75_rfpo_2gpu}"
export VIVIDEX_RM75_URDF_OVERRIDE="${REPO}/assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_driven12.urdf"

RM75_TRAJ="norm_trajectories/rm75_inspire_right/${SEQ}.npz"
python tools/retarget_rm75_inspire_reference.py \
  --src "norm_trajectories/${SEQ}.npz" \
  --dst "${RM75_TRAJ}" \
  --overwrite

launch_rm75_rfpo() {
  local gpu="$1"
  local run="$2"
  local lr="$3"
  local clip="$4"
  local arm_scale="$5"
  local hand_scale="$6"
  local hand_bias="$7"
  local close_reward="$8"
  local open_penalty="$9"
  local residual_scale="${10}"
  local noise="${11}"
  local dynamic_bias="${12}"
  local dynamic_start="${13}"
  local dynamic_full="${14}"
  local min_approach_reward="${15}"
  local palm_approach_reward="${16}"
  local contact_only="${17}"
  local requires_contact="${18}"
  local contact_boost="${19}"
  local early_close_penalty="${20}"
  local synergy_reward="${21}"
  local synergy_penalty="${22}"
  local bias_weights="${23}"
  local precontact_cap="${24}"

  echo "[rm75-rfpo-2gpu] launch ${run} on GPU${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash scripts/server/train_state_single.sh "${SEQ}" "${run}" \
      "agent=fpo" \
      "env.robot_name=rm75_inspire_right" \
      "env.norm_traj=True" \
      "+env.task_kwargs.rm75_arm_action_scale=${arm_scale}" \
      "+env.task_kwargs.rm75_hand_action_scale=${hand_scale}" \
      "+env.task_kwargs.rm75_hand_close_bias=${hand_bias}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_bias=${dynamic_bias}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_start=${dynamic_start}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_full=${dynamic_full}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_requires_contact=${requires_contact}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_contact_boost=${contact_boost}" \
      "+env.task_kwargs.rm75_hand_precontact_close_cap=${precontact_cap}" \
      "+env.task_kwargs.rm75_hand_close_bias_weights=${bias_weights}" \
      "+env.task_kwargs.rm75_hand_close_palm_dist=0.24" \
      "+env.task_kwargs.rm75_hand_close_min_finger_dist=0.13" \
      "env.info_keywords=[pregrasp_success,pregrasp_steps,imitate_steps,hand_jpos_err,obj_com_err,obj_rot_err,obj_lift,contact_count,stable_grasp_contact,fingertip_obj_dist,min_fingertip_obj_dist,palm_obj_dist,hand_close_fraction,hand_thumb_close_fraction,hand_main_close_fraction,hand_pinky_close_fraction,object_xy_drift,object_speed,hand_mjpos_err,stage,control_error,obj_tgt_dist]" \
      "+env.task_kwargs.reward_kwargs.pregrasp_success_thresh=0.075" \
      "+env.task_kwargs.reward_kwargs.no_contact_grace_steps=45" \
      "+env.task_kwargs.reward_kwargs.obj_com_done_thresh=0.22" \
      "+env.task_kwargs.reward_kwargs.object_reward_requires_contact=True" \
      "+env.task_kwargs.reward_kwargs.no_contact_penalty=0.12" \
      "+env.task_kwargs.reward_kwargs.finger_approach_reward_scale=2.5" \
      "+env.task_kwargs.reward_kwargs.finger_approach_scale=28.0" \
      "+env.task_kwargs.reward_kwargs.min_finger_approach_reward_scale=${min_approach_reward}" \
      "+env.task_kwargs.reward_kwargs.min_finger_approach_scale=18.0" \
      "+env.task_kwargs.reward_kwargs.palm_approach_reward_scale=${palm_approach_reward}" \
      "+env.task_kwargs.reward_kwargs.palm_approach_scale=8.0" \
      "+env.task_kwargs.reward_kwargs.contact_reward_scale=0.5" \
      "+env.task_kwargs.reward_kwargs.thumb_contact_reward_scale=0.9" \
      "+env.task_kwargs.reward_kwargs.non_thumb_contact_reward_scale=0.5" \
      "+env.task_kwargs.reward_kwargs.stable_contact_bonus=2.0" \
      "+env.task_kwargs.reward_kwargs.required_non_thumb_contacts=1" \
      "env.task_kwargs.reward_kwargs.object_reward_scale=10.0" \
      "env.task_kwargs.reward_kwargs.obj_err_scale=50.0" \
      "+env.task_kwargs.reward_kwargs.hand_mimic_reward_scale=2.0" \
      "+env.task_kwargs.reward_kwargs.hand_close_reward_scale=${close_reward}" \
      "+env.task_kwargs.reward_kwargs.hand_close_near_dist=0.11" \
      "+env.task_kwargs.reward_kwargs.hand_close_palm_near_dist=0.24" \
      "+env.task_kwargs.reward_kwargs.hand_close_min_finger_near_dist=0.13" \
      "+env.task_kwargs.reward_kwargs.hand_close_contact_only=${contact_only}" \
      "+env.task_kwargs.reward_kwargs.hand_close_target=0.65" \
      "+env.task_kwargs.reward_kwargs.hand_open_penalty_scale=${open_penalty}" \
      "+env.task_kwargs.reward_kwargs.early_close_penalty_scale=${early_close_penalty}" \
      "+env.task_kwargs.reward_kwargs.early_close_palm_dist=0.20" \
      "+env.task_kwargs.reward_kwargs.early_close_min_finger_dist=0.10" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_reward_scale=${synergy_reward}" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_penalty_scale=${synergy_penalty}" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_thumb_target_ratio=0.75" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_pinky_target_ratio=0.80" \
      "+env.task_kwargs.reward_kwargs.object_xy_drift_free_thresh=0.018" \
      "+env.task_kwargs.reward_kwargs.object_xy_drift_penalty_scale=18.0" \
      "+env.task_kwargs.reward_kwargs.object_speed_penalty_scale=0.03" \
      "+env.task_kwargs.reward_kwargs.lift_reward_scale=30.0" \
      "+env.task_kwargs.reward_kwargs.lift_reward_cap=0.08" \
      "env.task_kwargs.reward_kwargs.lift_bonus_thresh=0.015" \
      "env.task_kwargs.reward_kwargs.lift_bonus_mag=4.0" \
      "agent.params.actor_objective=fpo" \
      "agent.params.trust_region_mode=aspo" \
      "agent.params.learning_rate=${lr}" \
      "agent.params.clip_range=${clip}" \
      "agent.params.log_ratio_scale=0.5" \
      "agent.params.cfm_diff_clip=2.0" \
      "agent.params.cfm_loss_clamp=20.0" \
      "agent.params.action_head_mode=flow_residual" \
      "agent.params.residual_action_scale=${residual_scale}" \
      "agent.params.rollout_action_noise_std=${noise}" \
      "agent.params.rollout_deterministic=False" \
      "agent.params.eval_deterministic=False" \
      "agent.params.n_steps=4096" \
      "agent.params.batch_size=512" \
      "agent.params.n_epochs=4" \
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
      "wandb.group=rm75_rfpo_scratch_2gpu" \
      "wandb.sweep_name_prefix=${run}"
  ) > "${LOG_ROOT}/${run}.out" 2>&1 &
  echo "$!" > "${LOG_ROOT}/${run}.pid"
}

launch_rm75_rfpo 0 "rm75_rfpo_scratch_${RM75_RUN_TAG}_aspo_arm022_preclose_800k_gpu0" 5.0e-6 0.10 0.22 1.0 0.05 4.5 0.8 0.06 0.010 0.62 0.24 0.080 2.8 1.8 False False 0.30 1.0 0.8 0.9 "[0.42,0.52,1.0,1.0,0.92,0.68]" 0.45
launch_rm75_rfpo 1 "rm75_rfpo_scratch_${RM75_RUN_TAG}_aspo_arm020_thumblate_800k_gpu1" 5.0e-6 0.10 0.20 1.0 0.045 4.2 0.9 0.055 0.010 0.58 0.23 0.075 2.8 1.8 False False 0.35 1.2 1.0 1.0 "[0.32,0.42,1.0,1.0,0.9,0.62]" 0.42

echo "[rm75-rfpo-2gpu] launched jobs:"
jobs -l
wait
