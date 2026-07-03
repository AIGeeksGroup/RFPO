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
export RM75_RUN_TAG="${RM75_RUN_TAG:-v8_envelope_close}"
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
  local arm_scale="$3"
  local dynamic_bias="$4"
  local dynamic_start="$5"
  local dynamic_full="$6"
  local precontact_cap="$7"
  local thumb_cap="$8"
  local pinky_cap="$9"
  local thumb_delay="${10}"
  local pinky_delay="${11}"
  local close_reward="${12}"
  local open_penalty="${13}"
  local early_close_penalty="${14}"
  local bias_weights="${15}"
  local residual_scale="${16}"
  local min_approach_reward="${17}"
  local palm_approach_reward="${18}"

  echo "[rm75-rfpo-v8] launch ${run} on GPU${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash scripts/server/train_state_single.sh "${SEQ}" "${run}" \
      "agent=fpo" \
      "env.robot_name=rm75_inspire_right" \
      "env.norm_traj=True" \
      "+env.task_kwargs.rm75_arm_action_scale=${arm_scale}" \
      "+env.task_kwargs.rm75_hand_action_scale=1.0" \
      "+env.task_kwargs.rm75_hand_close_bias=0.035" \
      "+env.task_kwargs.rm75_hand_dynamic_close_bias=${dynamic_bias}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_start=${dynamic_start}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_full=${dynamic_full}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_requires_contact=False" \
      "+env.task_kwargs.rm75_hand_dynamic_close_contact_boost=0.40" \
      "+env.task_kwargs.rm75_hand_precontact_close_cap=${precontact_cap}" \
      "+env.task_kwargs.rm75_hand_precontact_thumb_cap=${thumb_cap}" \
      "+env.task_kwargs.rm75_hand_precontact_pinky_cap=${pinky_cap}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_thumb_delay=${thumb_delay}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_pinky_delay=${pinky_delay}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_finger_mode=mean" \
      "+env.task_kwargs.rm75_hand_close_bias_weights=${bias_weights}" \
      "+env.task_kwargs.rm75_hand_close_palm_dist=0.24" \
      "+env.task_kwargs.rm75_hand_close_min_finger_dist=0.13" \
      "env.info_keywords=[pregrasp_success,pregrasp_steps,imitate_steps,hand_jpos_err,obj_com_err,obj_rot_err,obj_lift,contact_count,stable_grasp_contact,fingertip_obj_dist,min_fingertip_obj_dist,palm_obj_dist,hand_close_fraction,hand_thumb_close_fraction,hand_main_close_fraction,hand_pinky_close_fraction,object_xy_drift,object_speed,hand_mjpos_err,stage,control_error,obj_tgt_dist]" \
      "+env.task_kwargs.reward_kwargs.pregrasp_success_thresh=0.075" \
      "+env.task_kwargs.reward_kwargs.no_contact_grace_steps=45" \
      "+env.task_kwargs.reward_kwargs.obj_com_done_thresh=0.22" \
      "+env.task_kwargs.reward_kwargs.object_reward_requires_contact=True" \
      "+env.task_kwargs.reward_kwargs.no_contact_penalty=0.10" \
      "+env.task_kwargs.reward_kwargs.finger_approach_reward_scale=2.6" \
      "+env.task_kwargs.reward_kwargs.finger_approach_scale=28.0" \
      "+env.task_kwargs.reward_kwargs.min_finger_approach_reward_scale=${min_approach_reward}" \
      "+env.task_kwargs.reward_kwargs.min_finger_approach_scale=18.0" \
      "+env.task_kwargs.reward_kwargs.palm_approach_reward_scale=${palm_approach_reward}" \
      "+env.task_kwargs.reward_kwargs.palm_approach_scale=8.0" \
      "+env.task_kwargs.reward_kwargs.contact_reward_scale=0.6" \
      "+env.task_kwargs.reward_kwargs.thumb_contact_reward_scale=0.9" \
      "+env.task_kwargs.reward_kwargs.non_thumb_contact_reward_scale=0.55" \
      "+env.task_kwargs.reward_kwargs.stable_contact_bonus=2.2" \
      "+env.task_kwargs.reward_kwargs.required_non_thumb_contacts=1" \
      "env.task_kwargs.reward_kwargs.object_reward_scale=10.0" \
      "env.task_kwargs.reward_kwargs.obj_err_scale=50.0" \
      "+env.task_kwargs.reward_kwargs.hand_mimic_reward_scale=1.7" \
      "+env.task_kwargs.reward_kwargs.hand_close_reward_scale=${close_reward}" \
      "+env.task_kwargs.reward_kwargs.hand_close_near_dist=0.12" \
      "+env.task_kwargs.reward_kwargs.hand_close_palm_near_dist=0.24" \
      "+env.task_kwargs.reward_kwargs.hand_close_min_finger_near_dist=0.13" \
      "+env.task_kwargs.reward_kwargs.hand_close_contact_only=False" \
      "+env.task_kwargs.reward_kwargs.hand_close_target=0.62" \
      "+env.task_kwargs.reward_kwargs.hand_open_penalty_scale=${open_penalty}" \
      "+env.task_kwargs.reward_kwargs.early_close_penalty_scale=${early_close_penalty}" \
      "+env.task_kwargs.reward_kwargs.early_close_palm_dist=0.23" \
      "+env.task_kwargs.reward_kwargs.early_close_min_finger_dist=0.14" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_reward_scale=1.1" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_penalty_scale=0.75" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_thumb_target_ratio=0.62" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_pinky_target_ratio=0.72" \
      "+env.task_kwargs.reward_kwargs.object_xy_drift_free_thresh=0.018" \
      "+env.task_kwargs.reward_kwargs.object_xy_drift_penalty_scale=18.0" \
      "+env.task_kwargs.reward_kwargs.object_speed_penalty_scale=0.035" \
      "+env.task_kwargs.reward_kwargs.lift_reward_scale=32.0" \
      "+env.task_kwargs.reward_kwargs.lift_reward_cap=0.08" \
      "env.task_kwargs.reward_kwargs.lift_bonus_thresh=0.015" \
      "env.task_kwargs.reward_kwargs.lift_bonus_mag=4.0" \
      "agent.params.actor_objective=fpo" \
      "agent.params.trust_region_mode=aspo" \
      "agent.params.learning_rate=5.0e-6" \
      "agent.params.clip_range=0.10" \
      "agent.params.log_ratio_scale=0.5" \
      "agent.params.cfm_diff_clip=2.0" \
      "agent.params.cfm_loss_clamp=20.0" \
      "agent.params.action_head_mode=flow_residual" \
      "agent.params.residual_action_scale=${residual_scale}" \
      "agent.params.rollout_action_noise_std=0.010" \
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

launch_rm75_rfpo 0 "rm75_rfpo_scratch_${RM75_RUN_TAG}_aspo_mainwrap_800k_gpu0" \
  0.21 0.72 0.255 0.080 0.58 0.22 0.36 0.22 0.12 4.7 0.55 0.55 \
  "[0.30,0.38,1.0,1.0,0.94,0.58]" 0.060 2.8 1.8

launch_rm75_rfpo 1 "rm75_rfpo_scratch_${RM75_RUN_TAG}_aspo_contactfinish_800k_gpu1" \
  0.20 0.66 0.240 0.075 0.52 0.18 0.32 0.26 0.14 4.4 0.65 0.65 \
  "[0.26,0.34,1.0,1.0,0.92,0.54]" 0.055 3.0 1.9

echo "[rm75-rfpo-v8] launched jobs:"
jobs -l
wait
