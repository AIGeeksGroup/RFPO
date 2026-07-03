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
export RM75_RUN_TAG="${RM75_RUN_TAG:-v11_vividex_gate}"
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
  local gate_mode="$7"
  local precontact_cap="$8"
  local thumb_cap="$9"
  local pinky_cap="${10}"
  local contact_boost="${11}"
  local lock_alpha="${12}"
  local close_reward="${13}"
  local early_close_penalty="${14}"
  local balance_penalty="${15}"
  local stable_bonus="${16}"
  local hold_bonus="${17}"
  local residual_scale="${18}"
  local bias_weights="${19}"
  local object_hold_steps="${20}"
  local tilt_penalty="${21}"
  local tilt_free="${22}"
  local ang_penalty="${23}"
  local unstable_push_penalty="${24}"
  local bad_push_drift="${25}"
  local bad_push_tilt="${26}"

  echo "[rm75-rfpo-v11] launch ${run} on GPU${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash scripts/server/train_state_single.sh "${SEQ}" "${run}" \
      "agent=fpo" \
      "env.robot_name=rm75_inspire_right" \
      "env.norm_traj=True" \
      "+env.task_kwargs.rm75_arm_action_scale=${arm_scale}" \
      "+env.task_kwargs.rm75_hand_action_scale=1.0" \
      "+env.task_kwargs.rm75_hand_close_bias=0.026" \
      "+env.task_kwargs.rm75_hand_dynamic_close_bias=${dynamic_bias}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_start=${dynamic_start}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_full=${dynamic_full}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_requires_contact=False" \
      "+env.task_kwargs.rm75_hand_dynamic_close_contact_boost=${contact_boost}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_gate_mode=${gate_mode}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_warmup_steps=5" \
      "+env.task_kwargs.rm75_hand_dynamic_close_ramp_steps=14" \
      "+env.task_kwargs.rm75_hand_dynamic_close_alpha_smooth=0.45" \
      "+env.task_kwargs.rm75_hand_dynamic_close_drift_gate=0.030" \
      "+env.task_kwargs.rm75_hand_dynamic_close_contact_hold_steps=2" \
      "+env.task_kwargs.rm75_hand_dynamic_close_contact_lock_alpha=${lock_alpha}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_group_mode=ratio" \
      "+env.task_kwargs.rm75_hand_dynamic_close_thumb_ratio=0.72" \
      "+env.task_kwargs.rm75_hand_dynamic_close_pinky_ratio=0.78" \
      "+env.task_kwargs.rm75_hand_precontact_close_cap=${precontact_cap}" \
      "+env.task_kwargs.rm75_hand_precontact_thumb_cap=${thumb_cap}" \
      "+env.task_kwargs.rm75_hand_precontact_pinky_cap=${pinky_cap}" \
      "+env.task_kwargs.rm75_hand_dynamic_close_finger_mode=mean" \
      "+env.task_kwargs.rm75_hand_close_bias_weights=${bias_weights}" \
      "+env.task_kwargs.rm75_hand_close_palm_dist=0.235" \
      "+env.task_kwargs.rm75_hand_close_min_finger_dist=0.125" \
      "env.info_keywords=[pregrasp_success,pregrasp_steps,imitate_steps,hand_jpos_err,obj_com_err,obj_rot_err,obj_lift,contact_count,stable_grasp_contact,rm75_grasp_score,fingertip_obj_dist,min_fingertip_obj_dist,palm_obj_dist,hand_close_fraction,hand_thumb_close_fraction,hand_main_close_fraction,hand_pinky_close_fraction,dynamic_close_alpha,contact_hold_steps,object_xy_drift,object_speed,object_tilt_err,object_ang_speed,hand_mjpos_err,stage,control_error,obj_tgt_dist]" \
      "+env.task_kwargs.reward_kwargs.pregrasp_success_thresh=0.075" \
      "+env.task_kwargs.reward_kwargs.no_contact_grace_steps=58" \
      "+env.task_kwargs.reward_kwargs.obj_com_done_thresh=0.24" \
      "+env.task_kwargs.reward_kwargs.object_reward_requires_contact=True" \
      "+env.task_kwargs.reward_kwargs.object_reward_requires_stable_contact=False" \
      "+env.task_kwargs.reward_kwargs.object_reward_contact_hold_steps=${object_hold_steps}" \
      "+env.task_kwargs.reward_kwargs.no_contact_penalty=0.08" \
      "+env.task_kwargs.reward_kwargs.finger_approach_reward_scale=1.8" \
      "+env.task_kwargs.reward_kwargs.finger_approach_scale=24.0" \
      "+env.task_kwargs.reward_kwargs.min_finger_approach_reward_scale=1.3" \
      "+env.task_kwargs.reward_kwargs.min_finger_approach_scale=16.0" \
      "+env.task_kwargs.reward_kwargs.palm_approach_reward_scale=1.2" \
      "+env.task_kwargs.reward_kwargs.palm_approach_scale=7.0" \
      "+env.task_kwargs.reward_kwargs.contact_reward_scale=0.95" \
      "+env.task_kwargs.reward_kwargs.thumb_contact_reward_scale=1.35" \
      "+env.task_kwargs.reward_kwargs.non_thumb_contact_reward_scale=0.95" \
      "+env.task_kwargs.reward_kwargs.stable_contact_bonus=${stable_bonus}" \
      "+env.task_kwargs.reward_kwargs.contact_hold_bonus_scale=${hold_bonus}" \
      "+env.task_kwargs.reward_kwargs.contact_hold_bonus_steps=6" \
      "+env.task_kwargs.reward_kwargs.required_non_thumb_contacts=1" \
      "env.task_kwargs.reward_kwargs.object_reward_scale=9.5" \
      "env.task_kwargs.reward_kwargs.obj_err_scale=48.0" \
      "+env.task_kwargs.reward_kwargs.hand_mimic_reward_scale=1.0" \
      "+env.task_kwargs.reward_kwargs.hand_close_reward_scale=${close_reward}" \
      "+env.task_kwargs.reward_kwargs.hand_close_near_dist=0.112" \
      "+env.task_kwargs.reward_kwargs.hand_close_palm_near_dist=0.225" \
      "+env.task_kwargs.reward_kwargs.hand_close_min_finger_near_dist=0.118" \
      "+env.task_kwargs.reward_kwargs.hand_close_contact_only=False" \
      "+env.task_kwargs.reward_kwargs.hand_close_target=0.64" \
      "+env.task_kwargs.reward_kwargs.hand_open_penalty_scale=0.75" \
      "+env.task_kwargs.reward_kwargs.early_close_penalty_scale=${early_close_penalty}" \
      "+env.task_kwargs.reward_kwargs.early_close_palm_dist=0.215" \
      "+env.task_kwargs.reward_kwargs.early_close_min_finger_dist=0.125" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_reward_scale=1.4" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_penalty_scale=1.0" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_balance_penalty_scale=${balance_penalty}" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_min_main_close=0.18" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_thumb_target_ratio=0.72" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_pinky_target_ratio=0.78" \
      "+env.task_kwargs.reward_kwargs.object_xy_drift_free_thresh=0.014" \
      "+env.task_kwargs.reward_kwargs.object_xy_drift_penalty_scale=24.0" \
      "+env.task_kwargs.reward_kwargs.object_speed_penalty_scale=0.055" \
      "+env.task_kwargs.reward_kwargs.object_tilt_free_thresh=${tilt_free}" \
      "+env.task_kwargs.reward_kwargs.object_tilt_penalty_scale=${tilt_penalty}" \
      "+env.task_kwargs.reward_kwargs.object_ang_vel_penalty_scale=${ang_penalty}" \
      "+env.task_kwargs.reward_kwargs.unstable_push_penalty_scale=${unstable_push_penalty}" \
      "+env.task_kwargs.reward_kwargs.bad_push_done=True" \
      "+env.task_kwargs.reward_kwargs.bad_push_min_step=7" \
      "+env.task_kwargs.reward_kwargs.bad_push_drift_thresh=${bad_push_drift}" \
      "+env.task_kwargs.reward_kwargs.bad_push_tilt_thresh=${bad_push_tilt}" \
      "+env.task_kwargs.reward_kwargs.lift_reward_scale=58.0" \
      "+env.task_kwargs.reward_kwargs.lift_reward_cap=0.09" \
      "env.task_kwargs.reward_kwargs.lift_bonus_thresh=0.010" \
      "env.task_kwargs.reward_kwargs.lift_bonus_mag=7.0" \
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
      "agent.params.best_metric=eval/mean_rm75_grasp_score" \
      "agent.params.degrade_metric=eval/mean_rm75_grasp_score" \
      "agent.params.save_best_model=True" \
      "agent.params.rollback_to_best_on_degrade=True" \
      "agent.params.rollback_patience=4" \
      "agent.params.degrade_threshold=5.0" \
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

launch_rm75_rfpo 0 "rm75_rfpo_scratch_${RM75_RUN_TAG}_aspo_v5hold_tiltgate_800k_gpu0" \
  0.205 0.70 0.250 0.080 blend 0.50 0.22 0.35 0.48 0.68 3.4 0.95 0.80 4.0 2.2 0.058 \
  "[0.28,0.36,1.0,1.0,0.96,0.62]" 3 8.0 0.22 0.030 12.0 0.070 0.48

launch_rm75_rfpo 1 "rm75_rfpo_scratch_${RM75_RUN_TAG}_aspo_antipush_hold_800k_gpu1" \
  0.198 0.66 0.238 0.075 envelope 0.44 0.18 0.30 0.60 0.78 3.0 1.15 1.05 4.8 3.0 0.052 \
  "[0.30,0.38,1.0,1.0,0.98,0.66]" 4 13.0 0.18 0.055 20.0 0.055 0.40

echo "[rm75-rfpo-v11] launched jobs:"
jobs -l
wait
