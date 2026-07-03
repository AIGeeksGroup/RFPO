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
export RM75_RUN_TAG="${RM75_RUN_TAG:-v16_vividex_like}"
export WANDB_GROUP="${WANDB_GROUP:-rm75_rfpo_scratch_2gpu}"
export WANDB_PREFIX="${WANDB_PREFIX:-rm75_rfpo_2gpu}"
export VIVIDEX_RM75_URDF_OVERRIDE="${REPO}/assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_driven12.urdf"
export VIVIDEX_N_ENVS="${VIVIDEX_N_ENVS:-16}"
export VIVIDEX_N_EVAL_ENVS="${VIVIDEX_N_EVAL_ENVS:-4}"

RM75_TRAJ="norm_trajectories/rm75_inspire_right/${SEQ}.npz"
python tools/retarget_rm75_inspire_reference.py \
  --src "norm_trajectories/${SEQ}.npz" \
  --dst "${RM75_TRAJ}" \
  --overwrite

info_keys="[pregrasp_success,pregrasp_steps,imitate_steps,hand_jpos_err,obj_com_err,obj_rot_err,obj_lift,contact_count,stable_grasp_contact,rm75_grasp_score,fingertip_obj_dist,min_fingertip_obj_dist,palm_obj_dist,hand_close_fraction,hand_thumb_close_fraction,hand_main_close_fraction,hand_pinky_close_fraction,reference_close_fraction,hand_reference_close_error,pregrasp_safe_active,effective_arm_action_scale,effective_hand_action_scale,dynamic_close_alpha,contact_hold_steps,object_xy_drift,object_speed,object_tilt_err,object_ang_speed,hand_mjpos_err,stage,control_error,obj_tgt_dist]"

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
  "+env.task_kwargs.rm75_arm_action_scale=0.205"
  "+env.task_kwargs.rm75_hand_action_scale=1.0"
  "+env.task_kwargs.rm75_hand_close_palm_dist=0.235"
  "+env.task_kwargs.rm75_hand_close_min_finger_dist=0.125"
  "+env.task_kwargs.rm75_hand_dynamic_close_finger_mode=mean"
  "env.info_keywords=${info_keys}"
  "+env.task_kwargs.reward_kwargs.pregrasp_success_thresh=0.075"
  "+env.task_kwargs.reward_kwargs.no_contact_grace_steps=58"
  "+env.task_kwargs.reward_kwargs.obj_com_done_thresh=0.24"
  "+env.task_kwargs.reward_kwargs.controller_penalty_scale=1000.0"
  "+env.task_kwargs.reward_kwargs.action_penalty_scale=0.01"
  "+env.task_kwargs.reward_kwargs.reward_divisor=10.0"
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
  "agent.params.best_metric=eval/mean_rm75_grasp_score"
  "agent.params.degrade_metric=eval/mean_rm75_grasp_score"
  "agent.params.save_best_model=True"
  "agent.params.rollback_to_best_on_degrade=True"
  "agent.params.rollback_patience=4"
  "agent.params.degrade_threshold=5.0"
  "eval_n_episodes=25"
  "total_timesteps=800000"
  "eval_freq=50000"
  "save_freq=100000"
  "restore_checkpoint_freq=100000"
  "wandb.group=rm75_rfpo_scratch_2gpu"
)

launch() {
  local gpu="$1"
  local run="$2"
  shift 2
  echo "[rm75-rfpo-v16] launch ${run} on GPU${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash scripts/server/train_state_single.sh "${SEQ}" "${run}" \
      "${common_args[@]}" \
      "$@" \
      "wandb.sweep_name_prefix=${run}"
  ) > "${LOG_ROOT}/${run}.out" 2>&1 &
  echo "$!" > "${LOG_ROOT}/${run}.pid"
}

# Branch A: closest to the original ViViDex stage-2 reward.
# It deliberately removes most RM75-specific close/synergy/drift shaping so we
# can test whether our custom shaping was blocking the original learning signal.
launch 0 "rm75_rfpo_scratch_${RM75_RUN_TAG}_original_reward_800k_gpu0" \
  "+env.task_kwargs.rm75_hand_close_bias=0.0" \
  "+env.task_kwargs.rm75_hand_dynamic_close_bias=0.0" \
  "+env.task_kwargs.reward_kwargs.finger_approach_reward_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.min_finger_approach_reward_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.palm_approach_reward_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.contact_reward_scale=0.5" \
  "+env.task_kwargs.reward_kwargs.thumb_contact_reward_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.non_thumb_contact_reward_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.stable_contact_bonus=0.0" \
  "+env.task_kwargs.reward_kwargs.contact_hold_bonus_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.required_non_thumb_contacts=1" \
  "+env.task_kwargs.reward_kwargs.object_reward_requires_contact=False" \
  "+env.task_kwargs.reward_kwargs.object_reward_requires_stable_contact=False" \
  "+env.task_kwargs.reward_kwargs.object_reward_contact_hold_steps=0" \
  "env.task_kwargs.reward_kwargs.object_reward_scale=10.0" \
  "env.task_kwargs.reward_kwargs.obj_err_scale=50.0" \
  "+env.task_kwargs.reward_kwargs.obj_rot_term=0.1" \
  "+env.task_kwargs.reward_kwargs.hand_mimic_reward_scale=4.0" \
  "+env.task_kwargs.reward_kwargs.hand_mimic_scale=10.0" \
  "+env.task_kwargs.reward_kwargs.hand_close_reward_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.hand_open_penalty_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.reference_close_reward_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.reference_close_penalty_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.early_close_penalty_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.hand_synergy_reward_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.hand_synergy_penalty_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.hand_synergy_balance_penalty_scale=0.0" \
  "env.task_kwargs.reward_kwargs.lift_bonus_thresh=0.020" \
  "env.task_kwargs.reward_kwargs.lift_bonus_mag=2.5" \
  "+env.task_kwargs.reward_kwargs.lift_reward_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.object_xy_drift_penalty_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.object_speed_penalty_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.object_tilt_penalty_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.object_ang_vel_penalty_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.unstable_push_penalty_scale=0.0" \
  "+env.task_kwargs.reward_kwargs.bad_push_done=False" \
  "agent.params.residual_action_scale=0.060"

# Branch B: still simple, but forces useful reward to pass through stable
# thumb+finger contact. This tests whether RM75 needs a contact gate because its
# hand kinematics are less forgiving than the original ViViDex hand.
launch 1 "rm75_rfpo_scratch_${RM75_RUN_TAG}_stable_contact_gate_800k_gpu1" \
  "+env.task_kwargs.rm75_hand_close_bias=0.012" \
  "+env.task_kwargs.rm75_hand_dynamic_close_bias=0.74" \
  "+env.task_kwargs.rm75_hand_dynamic_close_start=0.245" \
  "+env.task_kwargs.rm75_hand_dynamic_close_full=0.078" \
  "+env.task_kwargs.rm75_hand_dynamic_close_requires_contact=False" \
  "+env.task_kwargs.rm75_hand_dynamic_close_gate_mode=envelope" \
  "+env.task_kwargs.rm75_hand_dynamic_close_warmup_steps=6" \
  "+env.task_kwargs.rm75_hand_dynamic_close_ramp_steps=18" \
  "+env.task_kwargs.rm75_hand_dynamic_close_alpha_smooth=0.42" \
  "+env.task_kwargs.rm75_hand_dynamic_close_contact_hold_steps=2" \
  "+env.task_kwargs.rm75_hand_dynamic_close_contact_lock_alpha=0.72" \
  "+env.task_kwargs.rm75_hand_dynamic_close_group_mode=ratio" \
  "+env.task_kwargs.rm75_hand_dynamic_close_thumb_ratio=0.74" \
  "+env.task_kwargs.rm75_hand_dynamic_close_pinky_ratio=0.80" \
  "+env.task_kwargs.rm75_hand_precontact_close_cap=0.50" \
  "+env.task_kwargs.rm75_hand_precontact_thumb_cap=0.24" \
  "+env.task_kwargs.rm75_hand_precontact_pinky_cap=0.38" \
  "+env.task_kwargs.rm75_hand_close_bias_weights=[0.30,0.38,1.0,1.0,0.96,0.64]" \
  "+env.task_kwargs.reward_kwargs.finger_approach_reward_scale=1.6" \
  "+env.task_kwargs.reward_kwargs.finger_approach_scale=24.0" \
  "+env.task_kwargs.reward_kwargs.min_finger_approach_reward_scale=0.9" \
  "+env.task_kwargs.reward_kwargs.min_finger_approach_scale=16.0" \
  "+env.task_kwargs.reward_kwargs.palm_approach_reward_scale=0.7" \
  "+env.task_kwargs.reward_kwargs.palm_approach_scale=7.0" \
  "+env.task_kwargs.reward_kwargs.contact_reward_scale=0.8" \
  "+env.task_kwargs.reward_kwargs.thumb_contact_reward_scale=1.2" \
  "+env.task_kwargs.reward_kwargs.non_thumb_contact_reward_scale=0.95" \
  "+env.task_kwargs.reward_kwargs.stable_contact_bonus=7.0" \
  "+env.task_kwargs.reward_kwargs.contact_hold_bonus_scale=3.5" \
  "+env.task_kwargs.reward_kwargs.contact_hold_bonus_steps=6" \
  "+env.task_kwargs.reward_kwargs.required_non_thumb_contacts=1" \
  "+env.task_kwargs.reward_kwargs.object_reward_requires_contact=True" \
  "+env.task_kwargs.reward_kwargs.object_reward_requires_stable_contact=True" \
  "+env.task_kwargs.reward_kwargs.object_reward_contact_hold_steps=2" \
  "+env.task_kwargs.reward_kwargs.no_contact_penalty=0.05" \
  "env.task_kwargs.reward_kwargs.object_reward_scale=10.0" \
  "env.task_kwargs.reward_kwargs.obj_err_scale=50.0" \
  "+env.task_kwargs.reward_kwargs.hand_mimic_reward_scale=2.0" \
  "+env.task_kwargs.reward_kwargs.hand_close_reward_scale=1.6" \
  "+env.task_kwargs.reward_kwargs.hand_close_contact_only=False" \
  "+env.task_kwargs.reward_kwargs.hand_open_penalty_scale=0.35" \
  "+env.task_kwargs.reward_kwargs.reference_close_reward_scale=0.8" \
  "+env.task_kwargs.reward_kwargs.reference_close_penalty_scale=1.0" \
  "+env.task_kwargs.reward_kwargs.reference_close_tolerance=0.12" \
  "+env.task_kwargs.reward_kwargs.reference_close_start_step=4" \
  "+env.task_kwargs.reward_kwargs.early_close_penalty_scale=0.45" \
  "+env.task_kwargs.reward_kwargs.hand_synergy_reward_scale=0.6" \
  "+env.task_kwargs.reward_kwargs.hand_synergy_penalty_scale=0.4" \
  "+env.task_kwargs.reward_kwargs.hand_synergy_balance_penalty_scale=0.25" \
  "env.task_kwargs.reward_kwargs.lift_bonus_thresh=0.006" \
  "env.task_kwargs.reward_kwargs.lift_bonus_mag=8.0" \
  "+env.task_kwargs.reward_kwargs.lift_reward_scale=70.0" \
  "+env.task_kwargs.reward_kwargs.lift_reward_cap=0.09" \
  "+env.task_kwargs.reward_kwargs.object_xy_drift_free_thresh=0.018" \
  "+env.task_kwargs.reward_kwargs.object_xy_drift_penalty_scale=8.0" \
  "+env.task_kwargs.reward_kwargs.object_speed_penalty_scale=0.02" \
  "+env.task_kwargs.reward_kwargs.object_tilt_free_thresh=0.34" \
  "+env.task_kwargs.reward_kwargs.object_tilt_penalty_scale=1.0" \
  "+env.task_kwargs.reward_kwargs.object_ang_vel_penalty_scale=0.005" \
  "+env.task_kwargs.reward_kwargs.unstable_push_penalty_scale=1.0" \
  "+env.task_kwargs.reward_kwargs.bad_push_done=False" \
  "agent.params.residual_action_scale=0.056"

echo "[rm75-rfpo-v16] launched jobs:"
jobs -l
wait
