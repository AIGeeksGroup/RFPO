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
export RM75_RUN_TAG="${RM75_RUN_TAG:-v24_retarget_ablation}"
export WANDB_GROUP="${WANDB_GROUP:-rm75_rfpo_retarget_ablation_2gpu}"
export WANDB_PREFIX="${WANDB_PREFIX:-rm75_rfpo_retarget_ablation}"
export VIVIDEX_RM75_URDF_OVERRIDE="${REPO}/assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_driven12.urdf"
export VIVIDEX_N_ENVS="${VIVIDEX_N_ENVS:-16}"
export VIVIDEX_N_EVAL_ENVS="${VIVIDEX_N_EVAL_ENVS:-4}"

RM75_TRAJ_REACHABLE="norm_trajectories/rm75_inspire_right/${SEQ}.npz"
RM75_TRAJ_VIVIDEX="norm_trajectories/rm75_inspire_right_vividex_aligned/${SEQ}.npz"
python tools/retarget_rm75_inspire_reference.py \
  --src "norm_trajectories/${SEQ}.npz" \
  --dst "${RM75_TRAJ_REACHABLE}" \
  --mode reachable \
  --overwrite
python tools/retarget_rm75_inspire_reference.py \
  --src "norm_trajectories/${SEQ}.npz" \
  --dst "${RM75_TRAJ_VIVIDEX}" \
  --mode vividex-aligned \
  --overwrite

INFO_KEYS="[pregrasp_success,pregrasp_steps,imitate_steps,hand_jpos_err,obj_com_err,obj_rot_err,obj_lift,contact_count,stable_grasp_contact,thumb_contact,non_thumb_contact_count,palm_contact,stable_contact_hold_steps,rm75_grasp_score,rm75_precision_score,rm75_task_score,fingertip_obj_dist,min_fingertip_obj_dist,palm_obj_dist,hand_close_fraction,hand_thumb_close_fraction,hand_main_close_fraction,hand_pinky_close_fraction,reference_close_fraction,hand_reference_close_error,pregrasp_safe_active,effective_arm_action_scale,effective_hand_action_scale,project_to_object_active,dynamic_close_alpha,dynamic_close_thumb_alpha,dynamic_close_main_alpha,dynamic_close_pinky_alpha,contact_hold_steps,object_xy_drift,object_speed,object_tilt_err,object_ang_speed,hand_mjpos_err,stage,control_error,obj_tgt_dist]"

common_args=(
  "agent=fpo"
  "env.robot_name=rm75_inspire_right"
  "env.norm_traj=True"
  "+env.task_kwargs.rm75_pregrasp_safe_action=True"
  "+env.task_kwargs.rm75_pregrasp_safe_margin_steps=0"
  "+env.task_kwargs.rm75_pregrasp_arm_action_scale=0.0"
  "+env.task_kwargs.rm75_pregrasp_hand_action_scale=0.0"
  "+env.task_kwargs.rm75_pregrasp_hand_close_bias=0.0"
  "+env.task_kwargs.rm75_pregrasp_zero_hand_action=True"
  "+env.task_kwargs.rm75_arm_action_scale=0.155"
  "+env.task_kwargs.rm75_hand_action_scale=1.0"
  "+env.task_kwargs.rm75_post_pregrasp_arm_ramp_steps=24"
  "+env.task_kwargs.rm75_post_pregrasp_arm_scale_start=0.008"
  "+env.task_kwargs.rm75_post_pregrasp_arm_scale_end=0.155"
  "+env.task_kwargs.rm75_post_pregrasp_hold_until_stable=True"
  "+env.task_kwargs.rm75_post_pregrasp_hold_arm_scale=0.030"
  "+env.task_kwargs.rm75_post_pregrasp_required_stable_hold_steps=4"
  "+env.task_kwargs.rm75_post_pregrasp_project_to_object=True"
  "+env.task_kwargs.rm75_post_pregrasp_project_until_stable=True"
  "+env.task_kwargs.rm75_post_pregrasp_project_tangent_scale=0.16"
  "+env.task_kwargs.rm75_post_pregrasp_project_max_approach_speed=0.050"
  "+env.task_kwargs.rm75_post_pregrasp_project_max_retreat_speed=0.010"
  "+env.task_kwargs.rm75_hand_close_bias=0.016"
  "+env.task_kwargs.rm75_hand_close_palm_dist=0.235"
  "+env.task_kwargs.rm75_hand_close_min_finger_dist=0.125"
  "+env.task_kwargs.rm75_hand_dynamic_close_requires_contact=False"
  "+env.task_kwargs.rm75_hand_dynamic_close_gate_mode=envelope"
  "+env.task_kwargs.rm75_hand_dynamic_close_warmup_steps=5"
  "+env.task_kwargs.rm75_hand_dynamic_close_ramp_steps=20"
  "+env.task_kwargs.rm75_hand_dynamic_close_alpha_smooth=0.55"
  "+env.task_kwargs.rm75_hand_dynamic_close_drift_gate=0.022"
  "+env.task_kwargs.rm75_hand_dynamic_close_contact_hold_steps=2"
  "+env.task_kwargs.rm75_hand_dynamic_close_contact_lock_alpha=0.88"
  "+env.task_kwargs.rm75_hand_dynamic_close_phase_weight=0.35"
  "+env.task_kwargs.rm75_hand_dynamic_close_phase_start=2"
  "+env.task_kwargs.rm75_hand_dynamic_close_phase_ramp=26"
  "+env.task_kwargs.rm75_hand_dynamic_close_phase_near_gate=0.25"
  "+env.task_kwargs.rm75_hand_dynamic_close_group_mode=ratio"
  "+env.task_kwargs.rm75_hand_dynamic_close_thumb_ratio=0.78"
  "+env.task_kwargs.rm75_hand_dynamic_close_pinky_ratio=0.82"
  "+env.task_kwargs.rm75_hand_dynamic_close_finger_mode=mean"
  "+env.task_kwargs.rm75_hand_close_bias_weights=[0.30,0.40,1.0,1.0,0.98,0.66]"
  "env.info_keywords=${INFO_KEYS}"
  "+env.task_kwargs.reward_kwargs.pregrasp_success_thresh=0.075"
  "+env.task_kwargs.reward_kwargs.obj_com_done_thresh=0.145"
  "+env.task_kwargs.reward_kwargs.no_contact_grace_steps=10"
  "+env.task_kwargs.reward_kwargs.controller_penalty_scale=1000.0"
  "+env.task_kwargs.reward_kwargs.action_penalty_scale=0.01"
  "+env.task_kwargs.reward_kwargs.reward_divisor=10.0"
  "+env.task_kwargs.reward_kwargs.finger_approach_reward_scale=0.9"
  "+env.task_kwargs.reward_kwargs.finger_approach_scale=20.0"
  "+env.task_kwargs.reward_kwargs.min_finger_approach_reward_scale=0.7"
  "+env.task_kwargs.reward_kwargs.min_finger_approach_scale=14.0"
  "+env.task_kwargs.reward_kwargs.palm_approach_reward_scale=0.45"
  "+env.task_kwargs.reward_kwargs.palm_approach_scale=7.0"
  "+env.task_kwargs.reward_kwargs.contact_reward_scale=0.55"
  "+env.task_kwargs.reward_kwargs.thumb_contact_reward_scale=1.45"
  "+env.task_kwargs.reward_kwargs.non_thumb_contact_reward_scale=1.05"
  "+env.task_kwargs.reward_kwargs.required_non_thumb_contacts=1"
  "+env.task_kwargs.reward_kwargs.object_reward_requires_contact=True"
  "+env.task_kwargs.reward_kwargs.object_reward_requires_stable_contact=True"
  "+env.task_kwargs.reward_kwargs.object_reward_contact_hold_steps=4"
  "+env.task_kwargs.reward_kwargs.no_contact_penalty=0.22"
  "env.task_kwargs.reward_kwargs.object_reward_scale=8.0"
  "env.task_kwargs.reward_kwargs.obj_err_scale=50.0"
  "+env.task_kwargs.reward_kwargs.obj_rot_term=0.1"
  "+env.task_kwargs.reward_kwargs.hand_mimic_reward_scale=2.2"
  "+env.task_kwargs.reward_kwargs.hand_mimic_scale=10.0"
  "+env.task_kwargs.reward_kwargs.hand_close_reward_scale=2.2"
  "+env.task_kwargs.reward_kwargs.hand_close_near_dist=0.112"
  "+env.task_kwargs.reward_kwargs.hand_close_palm_near_dist=0.225"
  "+env.task_kwargs.reward_kwargs.hand_close_min_finger_near_dist=0.118"
  "+env.task_kwargs.reward_kwargs.hand_close_contact_only=False"
  "+env.task_kwargs.reward_kwargs.hand_close_target=0.68"
  "+env.task_kwargs.reward_kwargs.hand_open_penalty_scale=1.1"
  "+env.task_kwargs.reward_kwargs.reference_close_reward_scale=0.80"
  "+env.task_kwargs.reward_kwargs.reference_close_penalty_scale=1.00"
  "+env.task_kwargs.reward_kwargs.reference_close_tolerance=0.12"
  "+env.task_kwargs.reward_kwargs.reference_close_start_step=4"
  "+env.task_kwargs.reward_kwargs.early_close_penalty_scale=1.25"
  "+env.task_kwargs.reward_kwargs.early_close_palm_dist=0.215"
  "+env.task_kwargs.reward_kwargs.early_close_min_finger_dist=0.125"
  "+env.task_kwargs.reward_kwargs.hand_synergy_reward_scale=1.0"
  "+env.task_kwargs.reward_kwargs.hand_synergy_penalty_scale=0.80"
  "+env.task_kwargs.reward_kwargs.hand_synergy_balance_penalty_scale=0.75"
  "+env.task_kwargs.reward_kwargs.hand_synergy_min_main_close=0.18"
  "+env.task_kwargs.reward_kwargs.hand_synergy_thumb_target_ratio=0.78"
  "+env.task_kwargs.reward_kwargs.hand_synergy_pinky_target_ratio=0.82"
  "+env.task_kwargs.reward_kwargs.stable_contact_bonus=9.0"
  "+env.task_kwargs.reward_kwargs.contact_hold_bonus_scale=5.0"
  "+env.task_kwargs.reward_kwargs.contact_hold_bonus_steps=6"
  "env.task_kwargs.reward_kwargs.lift_bonus_thresh=0.010"
  "env.task_kwargs.reward_kwargs.lift_bonus_mag=6.0"
  "+env.task_kwargs.reward_kwargs.lift_reward_scale=42.0"
  "+env.task_kwargs.reward_kwargs.lift_reward_cap=0.09"
  "+env.task_kwargs.reward_kwargs.object_xy_drift_free_thresh=0.012"
  "+env.task_kwargs.reward_kwargs.object_xy_drift_penalty_scale=24.0"
  "+env.task_kwargs.reward_kwargs.object_speed_penalty_scale=0.060"
  "+env.task_kwargs.reward_kwargs.object_tilt_free_thresh=0.20"
  "+env.task_kwargs.reward_kwargs.object_tilt_penalty_scale=8.0"
  "+env.task_kwargs.reward_kwargs.object_ang_vel_penalty_scale=0.040"
  "+env.task_kwargs.reward_kwargs.unstable_push_penalty_scale=18.0"
  "+env.task_kwargs.reward_kwargs.bad_push_done=True"
  "+env.task_kwargs.reward_kwargs.bad_push_min_step=8"
  "+env.task_kwargs.reward_kwargs.bad_push_drift_thresh=0.048"
  "+env.task_kwargs.reward_kwargs.bad_push_tilt_thresh=0.36"
  "agent.params.actor_objective=fpo"
  "agent.params.trust_region_mode=aspo"
  "agent.params.learning_rate=5.0e-6"
  "agent.params.clip_range=0.10"
  "agent.params.log_ratio_scale=0.5"
  "agent.params.cfm_diff_clip=2.0"
  "agent.params.cfm_loss_clamp=20.0"
  "agent.params.action_head_mode=flow_residual"
  "agent.params.rollout_action_noise_std=0.010"
  "agent.params.rollout_deterministic=False"
  "agent.params.eval_deterministic=False"
  "agent.params.n_steps=4096"
  "agent.params.batch_size=512"
  "agent.params.n_epochs=4"
  "agent.params.n_samples_per_action=16"
  "agent.params.sampling_steps=8"
  "agent.params.actor_hidden_dims=[512,512]"
  "agent.params.critic_hidden_dims=[512,512]"
  "agent.params.activation=elu"
  "agent.params.max_grad_norm=1.0"
  "agent.params.actor_max_grad_norm=0.5"
  "agent.params.critic_max_grad_norm=2.0"
  "agent.params.curriculum_pregrasp_threshold=0.95"
  "agent.params.curriculum_stage2_metric=eval/mean_stable_grasp_contact"
  "agent.params.best_metric=eval/mean_rm75_task_score"
  "agent.params.degrade_metric=eval/mean_rm75_task_score"
  "agent.params.save_best_model=True"
  "agent.params.rollback_to_best_on_degrade=True"
  "agent.params.rollback_patience=4"
  "agent.params.degrade_threshold=5.0"
  "n_envs=16"
  "n_eval_envs=4"
  "eval_n_episodes=25"
  "total_timesteps=800000"
  "eval_freq=50000"
  "save_freq=100000"
  "restore_checkpoint_freq=100000"
  "wandb.group=rm75_rfpo_retarget_ablation_2gpu"
)

launch() {
  local gpu="$1"
  local run="$2"
  local traj_subdir="$3"
  shift 3
  echo "[rm75-rfpo-v24] launch ${run} on GPU${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    VIVIDEX_RM75_TRAJ_SUBDIR="${traj_subdir}" \
    bash scripts/server/train_state_single.sh "${SEQ}" "${run}" \
      "${common_args[@]}" \
      "$@" \
      "wandb.sweep_name_prefix=${run}"
  ) > "${LOG_ROOT}/${run}.out" 2>&1 &
  echo "$!" > "${LOG_ROOT}/${run}.pid"
}

launch 0 "rm75_rfpo_scratch_${RM75_RUN_TAG}_reachable_strict_800k_gpu0" \
  "rm75_inspire_right" \
  "agent.params.curriculum_stage2_threshold=0.045" \
  "++env.task_kwargs.rm75_arm_action_scale=0.150" \
  "++env.task_kwargs.rm75_post_pregrasp_arm_scale_end=0.150" \
  "++env.task_kwargs.rm75_post_pregrasp_hold_arm_scale=0.028" \
  "++env.task_kwargs.rm75_post_pregrasp_required_stable_hold_steps=5" \
  "++env.task_kwargs.rm75_post_pregrasp_project_tangent_scale=0.12" \
  "++env.task_kwargs.rm75_post_pregrasp_project_max_approach_speed=0.045" \
  "++env.task_kwargs.rm75_post_pregrasp_project_max_retreat_speed=0.008" \
  "++env.task_kwargs.rm75_hand_dynamic_close_bias=0.78" \
  "++env.task_kwargs.rm75_hand_dynamic_close_start=0.248" \
  "++env.task_kwargs.rm75_hand_dynamic_close_full=0.080" \
  "++env.task_kwargs.rm75_hand_precontact_close_cap=0.44" \
  "++env.task_kwargs.rm75_hand_precontact_thumb_cap=0.24" \
  "++env.task_kwargs.rm75_hand_precontact_pinky_cap=0.36" \
  "++env.task_kwargs.rm75_hand_dynamic_close_contact_boost=0.62" \
  "++env.task_kwargs.rm75_hand_dynamic_close_contact_lock_alpha=0.92" \
  "++env.task_kwargs.reward_kwargs.no_contact_grace_steps=8" \
  "++env.task_kwargs.reward_kwargs.object_reward_contact_hold_steps=5" \
  "++env.task_kwargs.reward_kwargs.bad_push_drift_thresh=0.044" \
  "++env.task_kwargs.reward_kwargs.bad_push_tilt_thresh=0.34" \
  "agent.params.residual_action_scale=0.042"

launch 1 "rm75_rfpo_scratch_${RM75_RUN_TAG}_vividex_aligned_v5wrap_800k_gpu1" \
  "rm75_inspire_right_vividex_aligned" \
  "agent.params.curriculum_stage2_threshold=0.040" \
  "++env.task_kwargs.rm75_arm_action_scale=0.165" \
  "++env.task_kwargs.rm75_post_pregrasp_arm_scale_end=0.165" \
  "++env.task_kwargs.rm75_post_pregrasp_hold_arm_scale=0.035" \
  "++env.task_kwargs.rm75_post_pregrasp_required_stable_hold_steps=4" \
  "++env.task_kwargs.rm75_post_pregrasp_project_tangent_scale=0.20" \
  "++env.task_kwargs.rm75_post_pregrasp_project_max_approach_speed=0.058" \
  "++env.task_kwargs.rm75_post_pregrasp_project_max_retreat_speed=0.012" \
  "++env.task_kwargs.rm75_hand_dynamic_close_bias=0.74" \
  "++env.task_kwargs.rm75_hand_dynamic_close_start=0.245" \
  "++env.task_kwargs.rm75_hand_dynamic_close_full=0.082" \
  "++env.task_kwargs.rm75_hand_precontact_close_cap=0.48" \
  "++env.task_kwargs.rm75_hand_precontact_thumb_cap=0.26" \
  "++env.task_kwargs.rm75_hand_precontact_pinky_cap=0.38" \
  "++env.task_kwargs.rm75_hand_dynamic_close_contact_boost=0.58" \
  "++env.task_kwargs.rm75_hand_dynamic_close_contact_lock_alpha=0.86" \
  "++env.task_kwargs.reward_kwargs.no_contact_grace_steps=12" \
  "++env.task_kwargs.reward_kwargs.object_reward_contact_hold_steps=4" \
  "++env.task_kwargs.reward_kwargs.bad_push_drift_thresh=0.052" \
  "++env.task_kwargs.reward_kwargs.bad_push_tilt_thresh=0.39" \
  "agent.params.residual_action_scale=0.048"

echo "[rm75-rfpo-v24] launched jobs:"
jobs -l
wait
