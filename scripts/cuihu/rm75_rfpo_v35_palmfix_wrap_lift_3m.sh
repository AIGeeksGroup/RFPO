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
export BASE_SEQ="${BASE_SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"
export WANDB_GROUP="${WANDB_GROUP:-rm75_rfpo_v35_palmfix_wrap_lift_3m}"
export WANDB_PREFIX="${WANDB_PREFIX:-rm75_v35}"
export VIVIDEX_RM75_URDF_OVERRIDE="${REPO}/assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_driven12.urdf"
export VIVIDEX_N_ENVS="${VIVIDEX_N_ENVS:-16}"
export VIVIDEX_N_EVAL_ENVS="${VIVIDEX_N_EVAL_ENVS:-4}"

RESULT_ROOT="${ROOT}/results/vividex_fpo_runtime/results/state_baseline"
V28A_RUN="${RESULT_ROOT}/rm75_rfpo_v28a_wrap_lift_v27v18c_last_3m_gpu0"
V28B_RUN="${RESULT_ROOT}/rm75_rfpo_v28b_palm_wrap_v27v5_best_3m_gpu1"
V28A_LAST="${V28A_RUN}/models/last.pt"
V28B_BEST="${V28B_RUN}/models/best.pt"
V35_CONTINUE_STEPS=3000000
V35_LANES="${V35_LANES:-0,1,2,3}"
V35_WAIT_CURRENT="${V35_WAIT_CURRENT:-1}"

INFO_KEYS="[pregrasp_success,pregrasp_steps,imitate_steps,hand_jpos_err,obj_com_err,obj_rot_err,obj_lift,contact_count,stable_grasp_contact,thumb_contact,non_thumb_contact_count,palm_contact,stable_contact_hold_steps,rm75_grasp_score,fingertip_obj_dist,min_fingertip_obj_dist,palm_obj_dist,hand_close_fraction,hand_thumb_close_fraction,hand_main_close_fraction,hand_pinky_close_fraction,reference_close_fraction,hand_reference_close_error,pregrasp_safe_active,effective_arm_action_scale,effective_hand_action_scale,dynamic_close_alpha,dynamic_close_thumb_alpha,dynamic_close_main_alpha,dynamic_close_pinky_alpha,contact_hold_steps,object_xy_drift,object_speed,object_tilt_err,object_ang_speed,hand_mjpos_err,stage,control_error,obj_tgt_dist,rm75_task_score]"

wait_for_current_rm75_training() {
  [[ "${V35_WAIT_CURRENT}" == "1" ]] || return 0
  echo "[rm75-v35] waiting for active v31/v33 RM75 training to finish"
  while ps -eo cmd | grep -E "python tools/train.py .*rm75_rfpo_v(31|33)" | grep -v grep >/dev/null; do
    ps -eo pid,etime,cmd | grep -E "python tools/train.py .*rm75_rfpo_v(31|33)" | grep -v grep | cut -c1-260 || true
    sleep 180
  done
}

should_run_lane() {
  local lane="$1"
  case ",${V35_LANES}," in
    *",${lane},"*) return 0 ;;
    *) return 1 ;;
  esac
}

checkpoint_step() {
  local ckpt="$1"
  local name
  name="$(basename "${ckpt}")"
  case "${name}" in
    fpo_step_*.pt) echo "${name#fpo_step_}" | sed "s/\.pt$//" ;;
    *) grep -E "total_timesteps:" "$(dirname "$(dirname "${ckpt}")")/exp_config.yaml" | tail -1 | awk "{print \$2}" ;;
  esac
}

require_checkpoint() {
  local label="$1"
  local ckpt="$2"
  [[ -n "${ckpt}" && -f "${ckpt}" ]] || {
    echo "[rm75-v35] missing checkpoint ${label}: ${ckpt}" >&2
    exit 1
  }
}

retarget_lane_seq() {
  local seq="$1"
  local ox="$2"
  local oy="$3"
  local oz="$4"
  local dst="norm_trajectories/rm75_inspire_right/${seq}.npz"
  (
    flock 9
    python tools/retarget_rm75_inspire_reference.py \
      --src "norm_trajectories/${BASE_SEQ}.npz" \
      --dst "${dst}" \
      --palm-offset "${ox}" "${oy}" "${oz}" \
      --overwrite
  ) 9>"${LOG_ROOT}/rm75_v35_retarget.lock"
}

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
  "+env.task_kwargs.rm75_arm_action_scale=0.30"
  "+env.task_kwargs.rm75_hand_action_scale=1.0"
  "+env.task_kwargs.rm75_post_pregrasp_arm_ramp_steps=10"
  "+env.task_kwargs.rm75_post_pregrasp_arm_scale_start=0.045"
  "+env.task_kwargs.rm75_post_pregrasp_arm_scale_end=0.30"
  "+env.task_kwargs.rm75_post_pregrasp_hold_until_stable=True"
  "+env.task_kwargs.rm75_post_pregrasp_hold_arm_scale=0.030"
  "+env.task_kwargs.rm75_post_pregrasp_required_stable_hold_steps=3"
  "+env.task_kwargs.rm75_post_pregrasp_max_hold_steps=8"
  "+env.task_kwargs.rm75_post_pregrasp_release_contact_hold_steps=1"
  "+env.task_kwargs.rm75_hand_close_bias=0.024"
  "+env.task_kwargs.rm75_hand_close_palm_dist=0.335"
  "+env.task_kwargs.rm75_hand_close_min_finger_dist=0.180"
  "+env.task_kwargs.rm75_hand_dynamic_close_start=0.345"
  "+env.task_kwargs.rm75_hand_dynamic_close_full=0.105"
  "+env.task_kwargs.rm75_hand_dynamic_close_requires_contact=False"
  "+env.task_kwargs.rm75_hand_dynamic_close_gate_mode=blend"
  "+env.task_kwargs.rm75_hand_dynamic_close_palm_weight=0.70"
  "+env.task_kwargs.rm75_hand_dynamic_close_finger_weight=0.30"
  "+env.task_kwargs.rm75_hand_dynamic_close_warmup_steps=0"
  "+env.task_kwargs.rm75_hand_dynamic_close_ramp_steps=10"
  "+env.task_kwargs.rm75_hand_dynamic_close_alpha_smooth=0.12"
  "+env.task_kwargs.rm75_hand_dynamic_close_drift_gate=0.070"
  "+env.task_kwargs.rm75_hand_dynamic_close_contact_hold_steps=1"
  "+env.task_kwargs.rm75_hand_dynamic_close_contact_lock_alpha=1.0"
  "+env.task_kwargs.rm75_hand_dynamic_close_contact_boost=0.95"
  "+env.task_kwargs.rm75_hand_dynamic_close_phase_weight=0.80"
  "+env.task_kwargs.rm75_hand_dynamic_close_phase_start=0"
  "+env.task_kwargs.rm75_hand_dynamic_close_phase_ramp=8"
  "+env.task_kwargs.rm75_hand_dynamic_close_phase_near_gate=0.345"
  "+env.task_kwargs.rm75_hand_dynamic_close_group_mode=ratio"
  "+env.task_kwargs.rm75_hand_dynamic_close_thumb_ratio=1.0"
  "+env.task_kwargs.rm75_hand_dynamic_close_pinky_ratio=1.0"
  "+env.task_kwargs.rm75_hand_dynamic_close_finger_mode=mean"
  "+env.task_kwargs.rm75_hand_precontact_close_cap=0.78"
  "+env.task_kwargs.rm75_hand_precontact_thumb_cap=0.72"
  "+env.task_kwargs.rm75_hand_precontact_pinky_cap=0.72"
  "+env.task_kwargs.rm75_hand_close_bias_weights=[1.0,1.0,1.08,1.08,1.08,1.02]"
  "env.info_keywords=${INFO_KEYS}"
  "+env.task_kwargs.reward_kwargs.pregrasp_success_thresh=0.087"
  "+env.task_kwargs.reward_kwargs.obj_com_done_thresh=0.220"
  "+env.task_kwargs.reward_kwargs.no_contact_grace_steps=26"
  "+env.task_kwargs.reward_kwargs.controller_penalty_scale=780.0"
  "+env.task_kwargs.reward_kwargs.action_penalty_scale=0.006"
  "+env.task_kwargs.reward_kwargs.reward_divisor=10.0"
  "+env.task_kwargs.reward_kwargs.finger_approach_reward_scale=0.60"
  "+env.task_kwargs.reward_kwargs.finger_approach_scale=11.0"
  "+env.task_kwargs.reward_kwargs.min_finger_approach_reward_scale=0.95"
  "+env.task_kwargs.reward_kwargs.min_finger_approach_scale=9.0"
  "+env.task_kwargs.reward_kwargs.palm_approach_reward_scale=4.2"
  "+env.task_kwargs.reward_kwargs.palm_approach_scale=4.8"
  "+env.task_kwargs.reward_kwargs.contact_reward_scale=0.18"
  "+env.task_kwargs.reward_kwargs.thumb_contact_reward_scale=2.8"
  "+env.task_kwargs.reward_kwargs.non_thumb_contact_reward_scale=3.3"
  "+env.task_kwargs.reward_kwargs.palm_contact_reward_scale=10.5"
  "+env.task_kwargs.reward_kwargs.required_non_thumb_contacts=2"
  "+env.task_kwargs.reward_kwargs.stable_contact_requires_palm=False"
  "+env.task_kwargs.reward_kwargs.object_reward_requires_contact=True"
  "+env.task_kwargs.reward_kwargs.object_reward_requires_stable_contact=False"
  "+env.task_kwargs.reward_kwargs.object_reward_contact_hold_steps=2"
  "+env.task_kwargs.reward_kwargs.no_contact_penalty=0.30"
  "env.task_kwargs.reward_kwargs.object_reward_scale=8.5"
  "env.task_kwargs.reward_kwargs.obj_err_scale=28.0"
  "+env.task_kwargs.reward_kwargs.obj_rot_term=0.05"
  "+env.task_kwargs.reward_kwargs.hand_mimic_reward_scale=1.4"
  "+env.task_kwargs.reward_kwargs.hand_mimic_scale=6.5"
  "+env.task_kwargs.reward_kwargs.hand_close_reward_scale=14.0"
  "+env.task_kwargs.reward_kwargs.hand_close_near_dist=0.180"
  "+env.task_kwargs.reward_kwargs.hand_close_palm_near_dist=0.340"
  "+env.task_kwargs.reward_kwargs.hand_close_min_finger_near_dist=0.185"
  "+env.task_kwargs.reward_kwargs.hand_close_contact_only=False"
  "+env.task_kwargs.reward_kwargs.hand_close_target=0.94"
  "+env.task_kwargs.reward_kwargs.hand_open_penalty_scale=18.0"
  "+env.task_kwargs.reward_kwargs.reference_close_reward_scale=0.8"
  "+env.task_kwargs.reward_kwargs.reference_close_penalty_scale=1.2"
  "+env.task_kwargs.reward_kwargs.reference_close_tolerance=0.14"
  "+env.task_kwargs.reward_kwargs.reference_close_start_step=0"
  "+env.task_kwargs.reward_kwargs.early_close_penalty_scale=0.12"
  "+env.task_kwargs.reward_kwargs.early_close_palm_dist=0.355"
  "+env.task_kwargs.reward_kwargs.early_close_min_finger_dist=0.190"
  "+env.task_kwargs.reward_kwargs.hand_synergy_reward_scale=2.4"
  "+env.task_kwargs.reward_kwargs.hand_synergy_penalty_scale=0.25"
  "+env.task_kwargs.reward_kwargs.hand_synergy_balance_penalty_scale=0.55"
  "+env.task_kwargs.reward_kwargs.hand_synergy_min_main_close=0.12"
  "+env.task_kwargs.reward_kwargs.hand_synergy_thumb_target_ratio=1.0"
  "+env.task_kwargs.reward_kwargs.hand_synergy_pinky_target_ratio=1.0"
  "+env.task_kwargs.reward_kwargs.lift_requires_full_wrap=True"
  "+env.task_kwargs.reward_kwargs.lift_wrap_min_quality=0.78"
  "+env.task_kwargs.reward_kwargs.lift_loose_wrap_penalty_scale=16.0"
  "+env.task_kwargs.reward_kwargs.no_lift_penalty_scale=45.0"
  "+env.task_kwargs.reward_kwargs.no_lift_penalty_min_step=8"
  "+env.task_kwargs.reward_kwargs.no_lift_penalty_lift_thresh=0.018"
  "+env.task_kwargs.reward_kwargs.no_lift_penalty_requires_lift_gate=False"
  "+env.task_kwargs.reward_kwargs.no_lift_penalty_contact_hold_steps=2"
  "+env.task_kwargs.reward_kwargs.half_wrap_penalty_scale=26.0"
  "+env.task_kwargs.reward_kwargs.half_wrap_penalty_contact_hold_steps=2"
  "+env.task_kwargs.reward_kwargs.half_wrap_penalty_lift_thresh=0.032"
  "+env.task_kwargs.reward_kwargs.half_wrap_required_non_thumb_contacts=2"
  "+env.task_kwargs.reward_kwargs.half_wrap_requires_thumb=True"
  "+env.task_kwargs.reward_kwargs.half_wrap_requires_palm=False"
  "+env.task_kwargs.reward_kwargs.half_wrap_min_close_quality=0.78"
  "+env.task_kwargs.reward_kwargs.stable_contact_bonus=15.0"
  "+env.task_kwargs.reward_kwargs.contact_hold_bonus_scale=11.0"
  "+env.task_kwargs.reward_kwargs.contact_hold_bonus_steps=8"
  "env.task_kwargs.reward_kwargs.lift_bonus_thresh=0.006"
  "env.task_kwargs.reward_kwargs.lift_bonus_mag=22.0"
  "+env.task_kwargs.reward_kwargs.lift_reward_scale=260.0"
  "+env.task_kwargs.reward_kwargs.lift_reward_cap=0.12"
  "+env.task_kwargs.reward_kwargs.object_xy_drift_free_thresh=0.034"
  "+env.task_kwargs.reward_kwargs.object_xy_drift_penalty_scale=6.2"
  "+env.task_kwargs.reward_kwargs.object_speed_penalty_scale=0.028"
  "+env.task_kwargs.reward_kwargs.object_tilt_free_thresh=0.36"
  "+env.task_kwargs.reward_kwargs.object_tilt_penalty_scale=3.0"
  "+env.task_kwargs.reward_kwargs.object_ang_vel_penalty_scale=0.018"
  "+env.task_kwargs.reward_kwargs.unstable_push_penalty_scale=6.0"
  "+env.task_kwargs.reward_kwargs.bad_push_done=False"
  "agent.params.actor_objective=fpo"
  "agent.params.trust_region_mode=aspo"
  "agent.params.learning_rate=2.3e-6"
  "agent.params.clip_range=0.10"
  "agent.params.log_ratio_scale=0.40"
  "agent.params.cfm_diff_clip=2.2"
  "agent.params.cfm_loss_clamp=20.0"
  "agent.params.action_head_mode=flow_residual"
  "agent.params.rollout_action_noise_std=0.018"
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
  "agent.params.resume_load_optimizer=False"
  "agent.params.curriculum_pregrasp_threshold=0.95"
  "agent.params.curriculum_stage2_metric=eval/mean_obj_lift"
  "agent.params.curriculum_stage2_threshold=0.008"
  "agent.params.save_best_model=True"
  "agent.params.rollback_to_best_on_degrade=True"
  "agent.params.rollback_patience=10"
  "agent.params.degrade_threshold=8.0"
  "n_envs=16"
  "n_eval_envs=4"
  "eval_n_episodes=25"
  "eval_freq=50000"
  "save_freq=100000"
  "restore_checkpoint_freq=100000"
  "wandb.group=${WANDB_GROUP}"
)

run_train() {
  local lane="$1"
  local gpu="$2"
  local run="$3"
  local resume="$4"
  local total="$5"
  local seq="$6"
  shift 6

  echo "[rm75-v35] launch lane=${lane} ${run} on GPU${gpu}, seq=${seq}, resume=${resume}, total=${total}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  bash scripts/server/train_state_single.sh "${seq}" "${run}" \
    "${common_args[@]}" \
    "resume_model=${resume}" \
    "total_timesteps=${total}" \
    "$@" \
    "wandb.sweep_name_prefix=${run}"
}

launch_lane() {
  local lane="$1"
  local step seq run ckpt
  case "${lane}" in
    0)
      ckpt="${V28A_LAST}"
      require_checkpoint "v28a-last" "${ckpt}"
      step="$(checkpoint_step "${ckpt}")"
      seq="${BASE_SEQ}_rm75_v35a_palmfix_lower_lift"
      retarget_lane_seq "${seq}" 0.052 -0.112 -0.015
      run="rm75_rfpo_v35a_palmfix_lower_lift_v28a_last_3m_gpu0"
      run_train 0 0 "${run}" "${ckpt}" "$((step + V35_CONTINUE_STEPS))" "${seq}" \
        "++env.task_kwargs.rm75_arm_action_scale=0.315" \
        "++env.task_kwargs.rm75_post_pregrasp_arm_scale_end=0.315" \
        "++env.task_kwargs.rm75_hand_dynamic_close_bias=1.65" \
        "++env.task_kwargs.reward_kwargs.lift_reward_scale=280.0" \
        "agent.params.best_metric=eval/mean_rm75_task_score" \
        "agent.params.degrade_metric=eval/mean_rm75_task_score" \
        "agent.params.residual_action_scale=0.090"
      ;;
    1)
      ckpt="${V28B_BEST}"
      require_checkpoint "v28b-best" "${ckpt}"
      step="$(checkpoint_step "${ckpt}")"
      seq="${BASE_SEQ}_rm75_v35b_palmfix_wrap_lock"
      retarget_lane_seq "${seq}" 0.060 -0.108 -0.020
      run="rm75_rfpo_v35b_palmfix_wrap_lock_v28b_best_3m_gpu1"
      run_train 1 1 "${run}" "${ckpt}" "$((step + V35_CONTINUE_STEPS))" "${seq}" \
        "++env.task_kwargs.rm75_arm_action_scale=0.270" \
        "++env.task_kwargs.rm75_post_pregrasp_required_stable_hold_steps=5" \
        "++env.task_kwargs.rm75_post_pregrasp_max_hold_steps=12" \
        "++env.task_kwargs.rm75_hand_dynamic_close_bias=1.95" \
        "++env.task_kwargs.reward_kwargs.palm_approach_reward_scale=4.8" \
        "++env.task_kwargs.reward_kwargs.palm_contact_reward_scale=12.0" \
        "agent.params.best_metric=eval/mean_rm75_grasp_score" \
        "agent.params.degrade_metric=eval/mean_rm75_grasp_score" \
        "agent.params.residual_action_scale=0.065"
      ;;
    2)
      ckpt="${V28A_LAST}"
      require_checkpoint "v28a-last" "${ckpt}"
      step="$(checkpoint_step "${ckpt}")"
      seq="${BASE_SEQ}_rm75_v35c_palmfix_reseat_lift"
      retarget_lane_seq "${seq}" 0.050 -0.116 -0.005
      run="rm75_rfpo_v35c_palmfix_reseat_lift_v28a_last_3m_gpu2"
      run_train 2 2 "${run}" "${ckpt}" "$((step + V35_CONTINUE_STEPS))" "${seq}" \
        "++env.task_kwargs.rm75_arm_action_scale=0.360" \
        "++env.task_kwargs.rm75_post_pregrasp_arm_ramp_steps=8" \
        "++env.task_kwargs.rm75_post_pregrasp_arm_scale_start=0.060" \
        "++env.task_kwargs.rm75_post_pregrasp_arm_scale_end=0.360" \
        "++env.task_kwargs.rm75_post_pregrasp_required_stable_hold_steps=2" \
        "++env.task_kwargs.rm75_post_pregrasp_max_hold_steps=7" \
        "++env.task_kwargs.rm75_hand_dynamic_close_bias=1.55" \
        "++env.task_kwargs.reward_kwargs.no_lift_penalty_scale=58.0" \
        "++env.task_kwargs.reward_kwargs.lift_reward_scale=340.0" \
        "agent.params.best_metric=eval/mean_obj_lift" \
        "agent.params.degrade_metric=eval/mean_obj_lift" \
        "agent.params.rollback_to_best_on_degrade=False" \
        "agent.params.residual_action_scale=0.115"
      ;;
    3)
      ckpt="${V28B_BEST}"
      require_checkpoint "v28b-best" "${ckpt}"
      step="$(checkpoint_step "${ckpt}")"
      seq="${BASE_SEQ}_rm75_v35d_palmfix_low_clamp"
      retarget_lane_seq "${seq}" 0.065 -0.110 -0.025
      run="rm75_rfpo_v35d_palmfix_low_clamp_v28b_best_3m_gpu3"
      run_train 3 3 "${run}" "${ckpt}" "$((step + V35_CONTINUE_STEPS))" "${seq}" \
        "++env.task_kwargs.rm75_arm_action_scale=0.250" \
        "++env.task_kwargs.rm75_post_pregrasp_required_stable_hold_steps=6" \
        "++env.task_kwargs.rm75_post_pregrasp_max_hold_steps=14" \
        "++env.task_kwargs.rm75_hand_dynamic_close_bias=2.20" \
        "++env.task_kwargs.reward_kwargs.hand_close_target=0.98" \
        "++env.task_kwargs.reward_kwargs.hand_open_penalty_scale=24.0" \
        "++env.task_kwargs.reward_kwargs.half_wrap_penalty_scale=38.0" \
        "++env.task_kwargs.reward_kwargs.half_wrap_required_non_thumb_contacts=3" \
        "++env.task_kwargs.reward_kwargs.half_wrap_requires_palm=True" \
        "agent.params.best_metric=eval/mean_rm75_grasp_score" \
        "agent.params.degrade_metric=eval/mean_rm75_grasp_score" \
        "agent.params.residual_action_scale=0.055"
      ;;
  esac
}

wait_for_current_rm75_training

for lane in 0 1 2 3; do
  if should_run_lane "${lane}"; then
    (
      launch_lane "${lane}"
    ) > "${LOG_ROOT}/rm75_rfpo_v35_gpu${lane}.out" 2>&1 &
    echo "$!" > "${LOG_ROOT}/rm75_rfpo_v35_gpu${lane}.pid"
  fi
done

echo "[rm75-v35] launched lanes (${V35_LANES}):"
jobs -l
wait
