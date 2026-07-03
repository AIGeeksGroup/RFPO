#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${VIVIDEX_VENV:-${REPO}/.venv-vividex}"
PYTHON="${VENV}/bin/python"
SEQ="${SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"
STAMP="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-rm75_native_rfpo_${STAMP}}"
RUN_ROOT="${VIVIDEX_LOCAL_RUN_ROOT:-${REPO}/.local_runs}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_NAME}}"
NATIVE_URDF="${NATIVE_URDF:-${REPO}/assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_nocyl.urdf}"
TRAJ_SUBDIR="${TRAJ_SUBDIR:-rm75_inspire_right_native12_clean_scale065_x090_y150_f080}"
TRAJ_PATH="${REPO}/norm_trajectories/${TRAJ_SUBDIR}/${SEQ}.npz"
RETARGET_MODE="${RETARGET_MODE:-reachable}"
RETARGET_PALM_OFFSET="${RETARGET_PALM_OFFSET:-0.090 -0.150 -0.002}"
OBJECT_SCALE="${OBJECT_SCALE:-${VIVIDEX_YCB_OBJECT_SCALE:-0.60}}"
RM75_BASE_OFFSET_LIST="${RM75_BASE_OFFSET_LIST:-0.0 0.0 0.0}"
RM75_BASE_RPY_LIST="${RM75_BASE_RPY_LIST:-0.0 0.0 0.0}"
RM75_BASE_YAW="${RM75_BASE_YAW:-0.0}"
USE_BC_ANCHOR="${USE_BC_ANCHOR:-1}"
BC_STAGE="${BC_STAGE:-0}"
BC_ANCHOR_DATASET="${BC_ANCHOR_DATASET:-${RUN_ROOT}/rm75_native_target_scale060_x090_y150_z119_f072_td2_pinky_m05_ref48.npz}"
BC_PINKY_CLOSE="${BC_PINKY_CLOSE-0.0}"
RETARGET_SEED_PINKY_QPOS="${RETARGET_SEED_PINKY_QPOS-0.03}"
DEFAULT_BC_CHECKPOINT="${RUN_ROOT}/rm75_native_target_scale060_x090_y150_z119_f072_td2_pinky_m05_ref48_direct_bc/flow_bc_last.pt"
if [[ "${NO_BC_CHECKPOINT:-0}" != "1" && -z "${BC_CHECKPOINT:-}" && -f "${DEFAULT_BC_CHECKPOINT}" ]]; then
  BC_CHECKPOINT="${DEFAULT_BC_CHECKPOINT}"
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Python env: ${PYTHON}" >&2
  echo "Run scripts/local/setup_vividex_venv.sh first." >&2
  exit 1
fi

cd "${REPO}"
mkdir -p "${RUN_ROOT}" "${RUN_DIR}" "$(dirname "${TRAJ_PATH}")"

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VIVIDEX_HEADLESS_NO_RENDER="${VIVIDEX_HEADLESS_NO_RENDER:-1}"
export VIVIDEX_RM75_URDF_OVERRIDE="${NATIVE_URDF}"
export VIVIDEX_RM75_TRAJ_SUBDIR="${TRAJ_SUBDIR}"
export VIVIDEX_RM75_HAND_DOF="${VIVIDEX_RM75_HAND_DOF:-12}"
export VIVIDEX_RM75_NATIVE_HAND_CONTROL=1
export VIVIDEX_RM75_ENFORCE_NATIVE_MIMIC="${VIVIDEX_RM75_ENFORCE_NATIVE_MIMIC:-0}"
export VIVIDEX_RM75_FINGER_STIFFNESS="${VIVIDEX_RM75_FINGER_STIFFNESS:-700}"
export VIVIDEX_RM75_FINGER_DAMPING="${VIVIDEX_RM75_FINGER_DAMPING:-90}"
export VIVIDEX_RM75_FINGER_FORCE_LIMIT="${VIVIDEX_RM75_FINGER_FORCE_LIMIT:-100}"
export VIVIDEX_RM75_STATIC_FRICTION="${VIVIDEX_RM75_STATIC_FRICTION:-2.5}"
export VIVIDEX_RM75_DYNAMIC_FRICTION="${VIVIDEX_RM75_DYNAMIC_FRICTION:-1.8}"
export VIVIDEX_RM75_DISABLE_PINKY_COLLISION="${VIVIDEX_RM75_DISABLE_PINKY_COLLISION:-0}"
export VIVIDEX_YCB_STATIC_FRICTION="${VIVIDEX_YCB_STATIC_FRICTION:-2.0}"
export VIVIDEX_YCB_DYNAMIC_FRICTION="${VIVIDEX_YCB_DYNAMIC_FRICTION:-1.5}"
export VIVIDEX_YCB_OBJECT_SCALE="${OBJECT_SCALE}"

if [[ ! -f "${TRAJ_PATH}" || "${REGENERATE_TRAJ:-0}" == "1" ]]; then
  RETARGET_CMD=(
    "${PYTHON}" tools/retarget_rm75_inspire_reference.py
    --src "norm_trajectories/${SEQ}.npz"
    --dst "${TRAJ_PATH}"
    --hand-dof 12
    --native-mimic-hand-qpos
    --mode "${RETARGET_MODE}"
    --overwrite
  )
  if [[ -n "${RETARGET_PALM_OFFSET}" ]]; then
    read -r -a PALM_OFFSET_ARGS <<< "${RETARGET_PALM_OFFSET}"
    RETARGET_CMD+=(--palm-offset "${PALM_OFFSET_ARGS[@]}")
  fi
  if [[ -n "${RETARGET_SEED_PINKY_QPOS:-}" ]]; then
    RETARGET_CMD+=(--seed-pinky-qpos "${RETARGET_SEED_PINKY_QPOS}")
  fi
  "${RETARGET_CMD[@]}"
fi

if [[ "${USE_BC_ANCHOR}" == "1" && ( ! -f "${BC_ANCHOR_DATASET}" || "${REGENERATE_BC_ANCHOR:-0}" == "1" ) ]]; then
  BC_FILTER_ARGS=()
  if [[ "${BC_FILTER_SUCCESS:-1}" == "1" ]]; then
    BC_FILTER_ARGS+=(--filter-success)
  fi
  BC_MIMIC_ARGS=()
  if [[ "${BC_ENFORCE_NATIVE_MIMIC_QPOS:-0}" == "1" ]]; then
    BC_MIMIC_ARGS+=(--enforce-native-mimic-qpos)
  fi
  BC_PINKY_ARGS=()
  if [[ -n "${BC_PINKY_CLOSE:-}" ]]; then
    BC_PINKY_ARGS+=(--pinky-close "${BC_PINKY_CLOSE}")
  fi
  read -r -a ROBOT_BASE_OFFSET_ARGS <<< "${RM75_BASE_OFFSET_LIST}"
  read -r -a ROBOT_BASE_RPY_ARGS <<< "${RM75_BASE_RPY_LIST}"
  "${PYTHON}" tools/collect_rm75_reference_bc_dataset.py \
    --output "${BC_ANCHOR_DATASET}" \
    --seq-name "${SEQ}" \
    --robot-name rm75_inspire_right \
    --norm-traj \
    --stage "${BC_STAGE}" \
    --mode approach_target \
    --native-hand-control \
    --native-apply-template-approach-pregrasp \
    "${BC_MIMIC_ARGS[@]}" \
    --robot-base-offset "${ROBOT_BASE_OFFSET_ARGS[@]}" \
    --robot-base-rpy "${ROBOT_BASE_RPY_ARGS[@]}" \
    --robot-base-yaw "${RM75_BASE_YAW}" \
    --pregrasp-success-mode proximity \
    --pregrasp-palm-dist-thresh 0.22 \
    --pregrasp-min-finger-dist-thresh 0.10 \
    --no-done-on-pregrasp-failure \
    --obj-com-done-thresh "${BC_OBJ_COM_DONE_THRESH:-0.35}" \
    --no-contact-grace-steps "${BC_NO_CONTACT_GRACE_STEPS:-260}" \
    --required-non-thumb-contacts "${BC_REQUIRED_NON_THUMB_CONTACTS:-2}" \
    --object-scale "${OBJECT_SCALE}" \
    --force-imitate-steps "${BC_FORCE_IMITATE_STEPS:-260}" \
    --num-trajs "${BC_ANCHOR_NUM_TRAJS:-48}" \
    --max-attempts "${BC_ANCHOR_MAX_ATTEMPTS:-512}" \
    "${BC_FILTER_ARGS[@]}" \
    --min-contact-steps "${BC_MIN_CONTACT_STEPS:-45}" \
    --min-stable-steps "${BC_MIN_STABLE_STEPS:-40}" \
    --min-thumb-steps "${BC_MIN_THUMB_STEPS:-45}" \
    --min-non-thumb-steps "${BC_MIN_NON_THUMB_STEPS:-45}" \
    --min-lift "${BC_MIN_LIFT:-0.08}" \
    --close-val "${BC_CLOSE_VAL:-0.30}" \
    --thumb-yaw "${BC_THUMB_YAW:-0.97}" \
    --thumb-pitch "${BC_THUMB_PITCH:-0.80}" \
    --finger-close "${BC_FINGER_CLOSE:-0.72}" \
    --thumb-delay-steps "${BC_THUMB_DELAY_STEPS:-2}" \
    --pinky-action-value "${BC_PINKY_ACTION_VALUE:--0.5}" \
    "${BC_PINKY_ARGS[@]}" \
    --approach-delta "${BC_APPROACH_DX:--0.0204}" "${BC_APPROACH_DY:--0.1595}" "${BC_APPROACH_DZ:-0.1186}" \
    --angular-action "${BC_ANGULAR_X:-0.0}" "${BC_ANGULAR_Y:-0.0}" "${BC_ANGULAR_Z:-0.0}" \
    --angular-start-step "${BC_ANGULAR_START_STEP:-80}" \
    --angular-end-step "${BC_ANGULAR_END_STEP:-205}" \
    --approach-steps "${BC_APPROACH_STEPS:-14}" \
    --close-steps "${BC_CLOSE_STEPS:-30}" \
    --hold-steps "${BC_HOLD_STEPS:-36}" \
    --lift-steps "${BC_LIFT_STEPS:-140}" \
    --lift-height "${BC_LIFT_HEIGHT:-0.18}" \
    --done-on-norm-success-10 \
    --norm-success-lift-thresh "${BC_NORM_SUCCESS_LIFT_THRESH:-0.08}" \
    --score-lift-target "${BC_SCORE_LIFT_TARGET:-0.08}" \
    --target-track-max-xy-step "${BC_TARGET_TRACK_MAX_XY_STEP:-0.006}" \
    --target-track-max-z-step "${BC_TARGET_TRACK_MAX_Z_STEP:-0.006}" \
    --target-track-gain "${BC_TARGET_TRACK_GAIN:-0.75}" \
    --target-track-z-deadband "${BC_TARGET_TRACK_Z_DEADBAND:-0.002}" \
    --settle-steps "${BC_SETTLE_STEPS:-80}"
fi

N_ENVS="${N_ENVS:-4}"
N_EVAL_ENVS="${N_EVAL_ENVS:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-800000}"
EVAL_FREQ="${EVAL_FREQ:-50000}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-12}"
SAVE_FREQ="${SAVE_FREQ:-100000}"
RESTORE_FREQ="${RESTORE_FREQ:-100000}"
MULTI_PROC="${MULTI_PROC:-False}"
N_STEPS="${N_STEPS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-512}"
N_EPOCHS="${N_EPOCHS:-4}"
N_SAMPLES="${N_SAMPLES:-16}"
ACTOR_DIMS="${ACTOR_DIMS:-[512,512]}"
CRITIC_DIMS="${CRITIC_DIMS:-[512,512]}"
RESIDUAL_DIMS="${RESIDUAL_DIMS:-[128,64]}"
DIRECT_DIMS="${DIRECT_DIMS:-[256,128]}"
RM75_BASE_OFFSET="${RM75_BASE_OFFSET:-[${RM75_BASE_OFFSET_LIST// /,}]}"
RM75_BASE_RPY="${RM75_BASE_RPY:-[${RM75_BASE_RPY_LIST// /,}]}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  N_ENVS="${SMOKE_N_ENVS:-1}"
  N_EVAL_ENVS="${SMOKE_N_EVAL_ENVS:-1}"
  TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS:-32}"
  EVAL_FREQ="${SMOKE_EVAL_FREQ:-64}"
  EVAL_N_EPISODES="${SMOKE_EVAL_N_EPISODES:-1}"
  SAVE_FREQ="${SMOKE_SAVE_FREQ:-64}"
  RESTORE_FREQ="${SMOKE_RESTORE_FREQ:-64}"
  MULTI_PROC=False
  N_STEPS="${SMOKE_N_STEPS:-8}"
  BATCH_SIZE="${SMOKE_BATCH_SIZE:-8}"
  N_EPOCHS="${SMOKE_N_EPOCHS:-1}"
  N_SAMPLES="${SMOKE_N_SAMPLES:-2}"
  if [[ -z "${BC_CHECKPOINT:-}" ]]; then
    ACTOR_DIMS="${SMOKE_ACTOR_DIMS:-[64,64]}"
    CRITIC_DIMS="${SMOKE_CRITIC_DIMS:-[64,64]}"
    RESIDUAL_DIMS="${SMOKE_RESIDUAL_DIMS:-[32,32]}"
    DIRECT_DIMS="${SMOKE_DIRECT_DIMS:-[64,64]}"
  fi
fi

INFO_KEYS="[pregrasp_success,pregrasp_steps,imitate_steps,hand_jpos_err,obj_com_err,obj_rot_err,obj_lift,contact_count,stable_grasp_contact,thumb_contact,non_thumb_contact_count,palm_contact,pinky_contact,ignore_pinky_contact,stable_contact_hold_steps,rm75_grasp_score,rm75_precision_score,rm75_task_score,fingertip_obj_dist,min_fingertip_obj_dist,palm_obj_dist,hand_close_fraction,hand_thumb_close_fraction,hand_main_close_fraction,hand_pinky_close_fraction,reference_close_fraction,hand_reference_close_error,pregrasp_safe_active,effective_arm_action_scale,effective_hand_action_scale,project_to_object_active,post_stable_lift_only_active,post_contact_lift_assist_active,post_contact_wrist_bias_active,scripted_lift_prior_active,lift_latch_steps_left,done_on_norm_success_10_active,contact_hold_steps,contact_success,lift_success_5cm,lift_success_target,norm_success_3,norm_success_10,object_xy_drift,object_xy_drift_x,object_xy_drift_y,object_speed,object_tilt_err,object_ang_speed,hand_mjpos_err,stage,control_error,obj_tgt_dist]"

CMD=(
  "${PYTHON}" tools/train.py
  agent=fpo
  "agent.multi_proc=${MULTI_PROC}"
  "env.name=${SEQ}"
  "env.robot_name=rm75_inspire_right"
  "env.norm_traj=True"
  "+env.task_kwargs.rm75_native_hand_control=True"
  "+env.task_kwargs.rm75_enforce_native_mimic_qpos=${RM75_ENFORCE_NATIVE_MIMIC_QPOS:-False}"
  "+env.task_kwargs.rm75_pregrasp_success_mode=proximity"
  "+env.task_kwargs.rm75_robot_base_offset=${RM75_BASE_OFFSET}"
  "+env.task_kwargs.rm75_robot_base_rpy=${RM75_BASE_RPY}"
  "+env.task_kwargs.rm75_robot_base_yaw=${RM75_BASE_YAW}"
  "+env.task_kwargs.rm75_pregrasp_palm_dist_thresh=0.22"
  "+env.task_kwargs.rm75_pregrasp_min_finger_dist_thresh=0.10"
  "+env.task_kwargs.rm75_done_on_pregrasp_failure=${RM75_DONE_ON_PREGRASP_FAILURE:-False}"
  "+env.task_kwargs.rm75_native_apply_template_approach_pregrasp=True"
  "+env.task_kwargs.object_scale=${OBJECT_SCALE}"
  "+env.task_kwargs.rm75_default_approach_delta=[${RM75_DEFAULT_APPROACH_DX:--0.0204},${RM75_DEFAULT_APPROACH_DY:--0.1595},${RM75_DEFAULT_APPROACH_DZ:-0.1186}]"
  "+env.task_kwargs.rm75_override_template_approach_delta=${RM75_OVERRIDE_TEMPLATE_APPROACH_DELTA:-True}"
  "+env.task_kwargs.rm75_follow_reference_approach_delta=${RM75_FOLLOW_REFERENCE_APPROACH_DELTA:-False}"
  "+env.task_kwargs.rm75_reset_settle_steps=${RM75_RESET_SETTLE_STEPS:-80}"
  "+env.task_kwargs.rm75_arm_action_scale=${RM75_ARM_ACTION_SCALE:-1.0}"
  "+env.task_kwargs.rm75_hand_action_scale=${RM75_HAND_ACTION_SCALE:-1.0}"
  "+env.task_kwargs.rm75_pregrasp_safe_action=${RM75_PREGRASP_SAFE_ACTION:-False}"
  "+env.task_kwargs.rm75_pregrasp_safe_margin_steps=${RM75_PREGRASP_SAFE_MARGIN_STEPS:-0}"
  "+env.task_kwargs.rm75_pregrasp_arm_action_scale=${RM75_PREGRASP_ARM_ACTION_SCALE:-1.0}"
  "+env.task_kwargs.rm75_pregrasp_hand_action_scale=${RM75_PREGRASP_HAND_ACTION_SCALE:-1.0}"
  "+env.task_kwargs.rm75_pregrasp_zero_hand_action=${RM75_PREGRASP_ZERO_HAND_ACTION:-False}"
  "+env.task_kwargs.rm75_post_pregrasp_arm_ramp_steps=${RM75_POST_PREGRASP_ARM_RAMP_STEPS:-0}"
  "+env.task_kwargs.rm75_post_pregrasp_arm_scale_start=${RM75_POST_PREGRASP_ARM_SCALE_START:-1.0}"
  "+env.task_kwargs.rm75_post_pregrasp_arm_scale_end=${RM75_POST_PREGRASP_ARM_SCALE_END:-1.0}"
  "+env.task_kwargs.rm75_post_pregrasp_hold_until_stable=${RM75_HOLD_UNTIL_STABLE:-False}"
  "+env.task_kwargs.rm75_post_pregrasp_hold_arm_scale=${RM75_HOLD_ARM_SCALE:-0.012}"
  "+env.task_kwargs.rm75_post_pregrasp_project_to_object=${RM75_PROJECT_TO_OBJECT:-False}"
  "+env.task_kwargs.rm75_post_pregrasp_project_until_stable=True"
  "+env.task_kwargs.rm75_post_pregrasp_project_tangent_scale=${RM75_PROJECT_TANGENT_SCALE:-0.02}"
  "+env.task_kwargs.rm75_post_pregrasp_project_max_approach_speed=${RM75_PROJECT_MAX_APPROACH_SPEED:-0.018}"
  "+env.task_kwargs.rm75_post_pregrasp_project_max_retreat_speed=${RM75_PROJECT_MAX_RETREAT_SPEED:-0.004}"
  "+env.task_kwargs.rm75_post_pregrasp_project_approach_bias=${RM75_PROJECT_APPROACH_BIAS:-0.006}"
  "+env.task_kwargs.rm75_post_pregrasp_required_stable_hold_steps=${RM75_REQUIRED_STABLE_HOLD_STEPS:-2}"
  "+env.task_kwargs.rm75_post_stable_lift_only=${RM75_POST_STABLE_LIFT_ONLY:-False}"
  "+env.task_kwargs.rm75_post_stable_max_xy_speed=${RM75_POST_STABLE_MAX_XY_SPEED:-0.012}"
  "+env.task_kwargs.rm75_post_stable_max_down_speed=${RM75_POST_STABLE_MAX_DOWN_SPEED:-0.002}"
  "+env.task_kwargs.rm75_post_stable_max_up_speed=${RM75_POST_STABLE_MAX_UP_SPEED:-0.035}"
  "+env.task_kwargs.rm75_post_stable_lift_bias=${RM75_POST_STABLE_LIFT_BIAS:-0.006}"
  "+env.task_kwargs.rm75_post_stable_max_angular_speed=${RM75_POST_STABLE_MAX_ANGULAR_SPEED:-0.10}"
  "+env.task_kwargs.rm75_post_stable_xy_drift_correction_gain=${RM75_POST_STABLE_XY_DRIFT_CORRECTION_GAIN:-0.0}"
  "+env.task_kwargs.rm75_post_stable_xy_drift_correction_max_speed=${RM75_POST_STABLE_XY_DRIFT_CORRECTION_MAX_SPEED:-0.0}"
  "+env.task_kwargs.rm75_post_lift_latch_steps=${RM75_POST_LIFT_LATCH_STEPS:-12}"
  "+env.task_kwargs.rm75_post_lift_latch_hand_alpha=${RM75_POST_LIFT_LATCH_HAND_ALPHA:-0.70}"
  "+env.task_kwargs.rm75_post_contact_lift_assist=${RM75_POST_CONTACT_LIFT_ASSIST:-False}"
  "+env.task_kwargs.rm75_post_contact_lift_min_hold_steps=${RM75_POST_CONTACT_LIFT_MIN_HOLD_STEPS:-2}"
  "+env.task_kwargs.rm75_post_contact_lift_min_non_thumb_contacts=${RM75_POST_CONTACT_LIFT_MIN_NON_THUMB_CONTACTS:-1}"
  "+env.task_kwargs.rm75_post_contact_lift_requires_thumb=${RM75_POST_CONTACT_LIFT_REQUIRES_THUMB:-True}"
  "+env.task_kwargs.rm75_post_contact_max_xy_speed=${RM75_POST_CONTACT_MAX_XY_SPEED:-0.004}"
  "+env.task_kwargs.rm75_post_contact_max_down_speed=${RM75_POST_CONTACT_MAX_DOWN_SPEED:-0.002}"
  "+env.task_kwargs.rm75_post_contact_max_up_speed=${RM75_POST_CONTACT_MAX_UP_SPEED:-0.045}"
  "+env.task_kwargs.rm75_post_contact_lift_bias=${RM75_POST_CONTACT_LIFT_BIAS:-0.012}"
  "+env.task_kwargs.rm75_post_contact_max_angular_speed=${RM75_POST_CONTACT_MAX_ANGULAR_SPEED:-0.08}"
  "+env.task_kwargs.rm75_post_contact_xy_drift_correction_gain=${RM75_POST_CONTACT_XY_DRIFT_CORRECTION_GAIN:-0.0}"
  "+env.task_kwargs.rm75_post_contact_xy_drift_correction_max_speed=${RM75_POST_CONTACT_XY_DRIFT_CORRECTION_MAX_SPEED:-0.0}"
  "+env.task_kwargs.rm75_post_contact_wrist_bias=${RM75_POST_CONTACT_WRIST_BIAS:-[0.0,0.0,0.0]}"
  "+env.task_kwargs.rm75_post_contact_wrist_min_hold_steps=${RM75_POST_CONTACT_WRIST_MIN_HOLD_STEPS:-2}"
  "+env.task_kwargs.rm75_post_contact_wrist_min_non_thumb_contacts=${RM75_POST_CONTACT_WRIST_MIN_NON_THUMB_CONTACTS:-1}"
  "+env.task_kwargs.rm75_post_contact_wrist_requires_thumb=${RM75_POST_CONTACT_WRIST_REQUIRES_THUMB:-True}"
  "+env.task_kwargs.rm75_scripted_lift_prior=${RM75_SCRIPTED_LIFT_PRIOR:-False}"
  "+env.task_kwargs.rm75_scripted_lift_prior_start_step=${RM75_SCRIPTED_LIFT_PRIOR_START_STEP:-0}"
  "+env.task_kwargs.rm75_scripted_lift_prior_approach_steps=${RM75_SCRIPTED_LIFT_PRIOR_APPROACH_STEPS:-14}"
  "+env.task_kwargs.rm75_scripted_lift_prior_close_steps=${RM75_SCRIPTED_LIFT_PRIOR_CLOSE_STEPS:-18}"
  "+env.task_kwargs.rm75_scripted_lift_prior_hold_steps=${RM75_SCRIPTED_LIFT_PRIOR_HOLD_STEPS:-10}"
  "+env.task_kwargs.rm75_scripted_lift_prior_lift_steps=${RM75_SCRIPTED_LIFT_PRIOR_LIFT_STEPS:-45}"
  "+env.task_kwargs.rm75_scripted_lift_prior_lift_height=${RM75_SCRIPTED_LIFT_PRIOR_LIFT_HEIGHT:-0.10}"
  "+env.task_kwargs.rm75_scripted_lift_prior_pos_gain=${RM75_SCRIPTED_LIFT_PRIOR_POS_GAIN:-0.04}"
  "+env.task_kwargs.rm75_scripted_lift_prior_blend=${RM75_SCRIPTED_LIFT_PRIOR_BLEND:-1.0}"
  "+env.task_kwargs.rm75_scripted_lift_prior_track_mode=${RM75_SCRIPTED_LIFT_PRIOR_TRACK_MODE:-initial}"
  "+env.task_kwargs.rm75_scripted_lift_prior_zero_angular=${RM75_SCRIPTED_LIFT_PRIOR_ZERO_ANGULAR:-True}"
  "+env.task_kwargs.rm75_scripted_lift_prior_follow_reference_approach=${RM75_SCRIPTED_LIFT_PRIOR_FOLLOW_REFERENCE_APPROACH:-False}"
  "+env.task_kwargs.rm75_scripted_hand_prior=${RM75_SCRIPTED_HAND_PRIOR:-False}"
  "+env.task_kwargs.rm75_scripted_hand_prior_blend=${RM75_SCRIPTED_HAND_PRIOR_BLEND:-1.0}"
  "+env.task_kwargs.rm75_scripted_hand_prior_approach_steps=${RM75_SCRIPTED_HAND_PRIOR_APPROACH_STEPS:-14}"
  "+env.task_kwargs.rm75_scripted_hand_prior_close_steps=${RM75_SCRIPTED_HAND_PRIOR_CLOSE_STEPS:-30}"
  "+env.task_kwargs.rm75_scripted_hand_prior_thumb_yaw=${RM75_SCRIPTED_HAND_PRIOR_THUMB_YAW:-0.90}"
  "+env.task_kwargs.rm75_scripted_hand_prior_thumb_pitch=${RM75_SCRIPTED_HAND_PRIOR_THUMB_PITCH:-0.85}"
  "+env.task_kwargs.rm75_scripted_hand_prior_finger_close=${RM75_SCRIPTED_HAND_PRIOR_FINGER_CLOSE:-0.75}"
  "+env.task_kwargs.rm75_scripted_hand_prior_pinky_close=${RM75_SCRIPTED_HAND_PRIOR_PINKY_CLOSE:-0.75}"
  "+env.task_kwargs.rm75_scripted_action_prior=${RM75_SCRIPTED_ACTION_PRIOR:-True}"
  "+env.task_kwargs.rm75_scripted_action_prior_blend=${RM75_SCRIPTED_ACTION_PRIOR_BLEND:-1.0}"
  "+env.task_kwargs.rm75_scripted_action_prior_approach_steps=${RM75_SCRIPTED_ACTION_PRIOR_APPROACH_STEPS:-14}"
  "+env.task_kwargs.rm75_scripted_action_prior_approach_z_power=${RM75_SCRIPTED_ACTION_PRIOR_APPROACH_Z_POWER:-1.0}"
  "+env.task_kwargs.rm75_scripted_action_prior_close_steps=${RM75_SCRIPTED_ACTION_PRIOR_CLOSE_STEPS:-30}"
  "+env.task_kwargs.rm75_scripted_action_prior_hold_steps=${RM75_SCRIPTED_ACTION_PRIOR_HOLD_STEPS:-36}"
  "+env.task_kwargs.rm75_scripted_action_prior_lift_steps=${RM75_SCRIPTED_ACTION_PRIOR_LIFT_STEPS:-140}"
  "+env.task_kwargs.rm75_scripted_action_prior_lift_height=${RM75_SCRIPTED_ACTION_PRIOR_LIFT_HEIGHT:-0.18}"
  "+env.task_kwargs.rm75_scripted_action_prior_pos_gain=${RM75_SCRIPTED_ACTION_PRIOR_POS_GAIN:-0.04}"
  "+env.task_kwargs.rm75_scripted_action_prior_mode=${RM75_SCRIPTED_ACTION_PRIOR_MODE:-target}"
  "+env.task_kwargs.rm75_scripted_action_prior_target_max_xy_step=${RM75_SCRIPTED_ACTION_PRIOR_TARGET_MAX_XY_STEP:-0.006}"
  "+env.task_kwargs.rm75_scripted_action_prior_target_max_z_step=${RM75_SCRIPTED_ACTION_PRIOR_TARGET_MAX_Z_STEP:-0.006}"
  "+env.task_kwargs.rm75_scripted_action_prior_target_gain=${RM75_SCRIPTED_ACTION_PRIOR_TARGET_GAIN:-0.75}"
  "+env.task_kwargs.rm75_scripted_action_prior_target_z_deadband=${RM75_SCRIPTED_ACTION_PRIOR_TARGET_Z_DEADBAND:-0.002}"
  "+env.task_kwargs.rm75_scripted_action_prior_target_pre_lift_height=${RM75_SCRIPTED_ACTION_PRIOR_TARGET_PRE_LIFT_HEIGHT:-0.0}"
  "+env.task_kwargs.rm75_scripted_action_prior_target_require_stable=${RM75_SCRIPTED_ACTION_PRIOR_TARGET_REQUIRE_STABLE:-False}"
  "+env.task_kwargs.rm75_scripted_action_prior_follow_reference_approach=${RM75_SCRIPTED_ACTION_PRIOR_FOLLOW_REFERENCE_APPROACH:-False}"
  "+env.task_kwargs.rm75_scripted_action_prior_thumb_yaw=${RM75_SCRIPTED_ACTION_PRIOR_THUMB_YAW:-0.97}"
  "+env.task_kwargs.rm75_scripted_action_prior_thumb_pitch=${RM75_SCRIPTED_ACTION_PRIOR_THUMB_PITCH:-0.80}"
  "+env.task_kwargs.rm75_scripted_action_prior_finger_close=${RM75_SCRIPTED_ACTION_PRIOR_FINGER_CLOSE:-0.72}"
  "+env.task_kwargs.rm75_scripted_action_prior_pinky_close=${RM75_SCRIPTED_ACTION_PRIOR_PINKY_CLOSE:-0.0}"
  "+env.task_kwargs.rm75_scripted_action_prior_grasp_offset=${RM75_SCRIPTED_ACTION_PRIOR_GRASP_OFFSET:-[0.0,0.0,0.0]}"
  "+env.task_kwargs.rm75_scripted_action_prior_offset_pregrasp=${RM75_SCRIPTED_ACTION_PRIOR_OFFSET_PREGRASP:-False}"
  "+env.task_kwargs.rm75_scripted_action_prior_thumb_delay_steps=${RM75_SCRIPTED_ACTION_PRIOR_THUMB_DELAY_STEPS:-2}"
  "+env.task_kwargs.rm75_scripted_action_prior_thumb_close_steps=${RM75_SCRIPTED_ACTION_PRIOR_THUMB_CLOSE_STEPS:-0}"
  "+env.task_kwargs.rm75_thumb_action_cap_until_non_thumb=${RM75_THUMB_ACTION_CAP_UNTIL_NON_THUMB:-null}"
  "+env.task_kwargs.rm75_thumb_action_cap_release_non_thumb_contacts=${RM75_THUMB_ACTION_CAP_RELEASE_NON_THUMB_CONTACTS:-2}"
  "+env.task_kwargs.rm75_thumb_action_cap_release_contact_hold_steps=${RM75_THUMB_ACTION_CAP_RELEASE_CONTACT_HOLD_STEPS:-1}"
  "+env.task_kwargs.rm75_hand_close_bias=${RM75_HAND_CLOSE_BIAS:-0.0}"
  "+env.task_kwargs.rm75_hand_phase_close_bias=${RM75_HAND_PHASE_CLOSE_BIAS:-0.0}"
  "+env.task_kwargs.rm75_hand_phase_close_start_step=${RM75_HAND_PHASE_CLOSE_START_STEP:-5}"
  "+env.task_kwargs.rm75_hand_phase_close_ramp_steps=${RM75_HAND_PHASE_CLOSE_RAMP_STEPS:-18}"
  "+env.task_kwargs.rm75_hand_dynamic_close_bias=${RM75_HAND_DYNAMIC_CLOSE_BIAS:-0.0}"
  "+env.task_kwargs.rm75_hand_dynamic_close_requires_contact=${RM75_HAND_DYNAMIC_CLOSE_REQUIRES_CONTACT:-True}"
  "+env.task_kwargs.rm75_hand_dynamic_close_start=0.30"
  "+env.task_kwargs.rm75_hand_dynamic_close_full=0.10"
  "+env.task_kwargs.rm75_hand_dynamic_close_finger_mode=min"
  "+env.task_kwargs.rm75_hand_dynamic_close_gate_mode=blend"
  "+env.task_kwargs.rm75_hand_dynamic_close_warmup_steps=0"
  "+env.task_kwargs.rm75_hand_dynamic_close_ramp_steps=10"
  "+env.task_kwargs.rm75_hand_dynamic_close_alpha_smooth=0.20"
  "+env.task_kwargs.rm75_hand_dynamic_close_contact_boost=0.40"
  "+env.task_kwargs.rm75_hand_dynamic_close_non_thumb_thumb_alpha=1.0"
  "+env.task_kwargs.rm75_hand_dynamic_close_contact_hold_steps=2"
  "+env.task_kwargs.rm75_hand_dynamic_close_contact_lock_alpha=1.0"
  "+env.task_kwargs.rm75_hand_dynamic_close_group_mode=ratio"
  "+env.task_kwargs.rm75_hand_dynamic_close_thumb_ratio=1.0"
  "+env.task_kwargs.rm75_hand_dynamic_close_pinky_ratio=${RM75_HAND_DYNAMIC_CLOSE_PINKY_RATIO:-0.0}"
  "+env.task_kwargs.rm75_hand_precontact_close_cap=1.0"
  "+env.task_kwargs.rm75_hand_precontact_thumb_cap=1.0"
  "+env.task_kwargs.rm75_hand_precontact_pinky_cap=${RM75_HAND_PRECONTACT_PINKY_CAP:-0.0}"
  "+env.task_kwargs.rm75_hand_close_bias_weights=${RM75_HAND_CLOSE_BIAS_WEIGHTS:-[2.8,2.2,1.0,1.0,1.0,0.0]}"
  "+env.task_kwargs.rm75_force_imitate_steps=${RM75_FORCE_IMITATE_STEPS:-260}"
  "+env.task_kwargs.rm75_done_on_norm_success_10=${RM75_DONE_ON_NORM_SUCCESS_10:-True}"
  "+env.task_kwargs.rm75_norm_success_lift_thresh=${RM75_NORM_SUCCESS_LIFT_THRESH:-0.08}"
  "+env.task_kwargs.rm75_norm_success_requires_contact=True"
  "+env.task_kwargs.rm75_success_min_non_thumb_contacts=${RM75_SUCCESS_MIN_NON_THUMB_CONTACTS:-2}"
  "+env.task_kwargs.rm75_score_lift_target=${RM75_SCORE_LIFT_TARGET:-0.08}"
  "+env.task_kwargs.rm75_ignore_pinky_contact=${RM75_IGNORE_PINKY_CONTACT:-True}"
  "+env.task_kwargs.rm75_disable_pinky_action=${RM75_DISABLE_PINKY_ACTION:-True}"
  "+env.task_kwargs.rm75_pinky_action_value=${RM75_PINKY_ACTION_VALUE:--0.5}"
  "env.info_keywords=${INFO_KEYS}"
  "+env.task_kwargs.reward_kwargs.pregrasp_success_thresh=0.085"
  "+env.task_kwargs.reward_kwargs.obj_com_done_thresh=${RM75_OBJ_COM_DONE_THRESH:-0.35}"
  "+env.task_kwargs.reward_kwargs.no_contact_grace_steps=${RM75_NO_CONTACT_GRACE_STEPS:-260}"
  "+env.task_kwargs.reward_kwargs.controller_penalty_scale=${RM75_CONTROLLER_PENALTY_SCALE:-700.0}"
  "+env.task_kwargs.reward_kwargs.action_penalty_scale=${RM75_ACTION_PENALTY_SCALE:-0.008}"
  "+env.task_kwargs.reward_kwargs.reward_divisor=${RM75_REWARD_DIVISOR:-10.0}"
  "+env.task_kwargs.reward_kwargs.finger_approach_reward_scale=${RM75_FINGER_APPROACH_REWARD_SCALE:-2.0}"
  "+env.task_kwargs.reward_kwargs.finger_approach_scale=${RM75_FINGER_APPROACH_SCALE:-22.0}"
  "+env.task_kwargs.reward_kwargs.min_finger_approach_reward_scale=${RM75_MIN_FINGER_APPROACH_REWARD_SCALE:-2.0}"
  "+env.task_kwargs.reward_kwargs.min_finger_approach_scale=${RM75_MIN_FINGER_APPROACH_SCALE:-18.0}"
  "+env.task_kwargs.reward_kwargs.palm_approach_reward_scale=${RM75_PALM_APPROACH_REWARD_SCALE:-0.0}"
  "+env.task_kwargs.reward_kwargs.palm_approach_scale=${RM75_PALM_APPROACH_SCALE:-7.0}"
  "+env.task_kwargs.reward_kwargs.contact_reward_scale=${RM75_CONTACT_REWARD_SCALE:-0.0}"
  "+env.task_kwargs.reward_kwargs.thumb_contact_reward_scale=${RM75_THUMB_CONTACT_REWARD_SCALE:-6.0}"
  "+env.task_kwargs.reward_kwargs.non_thumb_contact_reward_scale=${RM75_NON_THUMB_CONTACT_REWARD_SCALE:-6.0}"
  "+env.task_kwargs.reward_kwargs.required_non_thumb_contacts=${RM75_REQUIRED_NON_THUMB_CONTACTS:-2}"
  "+env.task_kwargs.reward_kwargs.object_reward_requires_contact=${RM75_OBJECT_REWARD_REQUIRES_CONTACT:-True}"
  "+env.task_kwargs.reward_kwargs.object_reward_requires_stable_contact=${RM75_OBJECT_REWARD_REQUIRES_STABLE_CONTACT:-True}"
  "+env.task_kwargs.reward_kwargs.object_reward_contact_hold_steps=${RM75_OBJECT_REWARD_CONTACT_HOLD_STEPS:-2}"
  "env.task_kwargs.reward_kwargs.object_reward_scale=${RM75_OBJECT_REWARD_SCALE:-9.0}"
  "env.task_kwargs.reward_kwargs.obj_err_scale=${RM75_OBJ_ERR_SCALE:-45.0}"
  "+env.task_kwargs.reward_kwargs.obj_rot_term=${RM75_OBJ_ROT_TERM:-0.1}"
  "+env.task_kwargs.reward_kwargs.hand_mimic_reward_scale=${RM75_HAND_MIMIC_REWARD_SCALE:-1.2}"
  "+env.task_kwargs.reward_kwargs.hand_mimic_scale=${RM75_HAND_MIMIC_SCALE:-8.0}"
  "+env.task_kwargs.reward_kwargs.hand_close_reward_scale=${RM75_HAND_CLOSE_REWARD_SCALE:-0.80}"
  "+env.task_kwargs.reward_kwargs.hand_open_penalty_scale=${RM75_HAND_OPEN_PENALTY_SCALE:-2.20}"
  "+env.task_kwargs.reward_kwargs.hand_close_target=${RM75_HAND_CLOSE_TARGET:-0.80}"
  "+env.task_kwargs.reward_kwargs.thumb_close_reward_scale=${RM75_THUMB_CLOSE_REWARD_SCALE:-0.0}"
  "+env.task_kwargs.reward_kwargs.thumb_open_penalty_scale=${RM75_THUMB_OPEN_PENALTY_SCALE:-0.0}"
  "+env.task_kwargs.reward_kwargs.thumb_close_target=${RM75_THUMB_CLOSE_TARGET:-0.55}"
  "+env.task_kwargs.reward_kwargs.reference_close_reward_scale=${RM75_REFERENCE_CLOSE_REWARD_SCALE:-0.25}"
  "+env.task_kwargs.reward_kwargs.reference_close_penalty_scale=${RM75_REFERENCE_CLOSE_PENALTY_SCALE:-1.20}"
  "+env.task_kwargs.reward_kwargs.reference_close_tolerance=${RM75_REFERENCE_CLOSE_TOLERANCE:-0.04}"
  "+env.task_kwargs.reward_kwargs.early_close_penalty_scale=${RM75_EARLY_CLOSE_PENALTY_SCALE:-0.15}"
  "+env.task_kwargs.reward_kwargs.hand_synergy_reward_scale=${RM75_HAND_SYNERGY_REWARD_SCALE:-0.40}"
  "+env.task_kwargs.reward_kwargs.hand_synergy_penalty_scale=${RM75_HAND_SYNERGY_PENALTY_SCALE:-0.20}"
  "+env.task_kwargs.reward_kwargs.hand_synergy_min_main_close=${RM75_HAND_SYNERGY_MIN_MAIN_CLOSE:-0.18}"
  "+env.task_kwargs.reward_kwargs.stable_contact_bonus=${RM75_STABLE_CONTACT_BONUS:-18.0}"
  "+env.task_kwargs.reward_kwargs.contact_hold_bonus_scale=${RM75_CONTACT_HOLD_BONUS_SCALE:-0.0}"
  "+env.task_kwargs.reward_kwargs.contact_hold_bonus_steps=${RM75_CONTACT_HOLD_BONUS_STEPS:-6}"
  "env.task_kwargs.reward_kwargs.lift_bonus_thresh=${RM75_LIFT_BONUS_THRESH:-0.012}"
  "env.task_kwargs.reward_kwargs.lift_bonus_mag=${RM75_LIFT_BONUS_MAG:-8.0}"
  "+env.task_kwargs.reward_kwargs.lift_reward_scale=${RM75_LIFT_REWARD_SCALE:-80.0}"
  "+env.task_kwargs.reward_kwargs.lift_reward_cap=${RM75_LIFT_REWARD_CAP:-0.08}"
  "+env.task_kwargs.reward_kwargs.object_xy_drift_free_thresh=${RM75_OBJECT_XY_DRIFT_FREE_THRESH:-0.018}"
  "+env.task_kwargs.reward_kwargs.object_xy_drift_penalty_scale=${RM75_OBJECT_XY_DRIFT_PENALTY_SCALE:-90.0}"
  "+env.task_kwargs.reward_kwargs.object_speed_penalty_scale=${RM75_OBJECT_SPEED_PENALTY_SCALE:-1.0}"
  "+env.task_kwargs.reward_kwargs.object_tilt_penalty_scale=${RM75_OBJECT_TILT_PENALTY_SCALE:-10.0}"
  "+env.task_kwargs.reward_kwargs.object_ang_vel_penalty_scale=${RM75_OBJECT_ANG_VEL_PENALTY_SCALE:-0.0}"
  "+env.task_kwargs.reward_kwargs.unstable_push_penalty_scale=${RM75_UNSTABLE_PUSH_PENALTY_SCALE:-35.0}"
  "+env.task_kwargs.reward_kwargs.bad_push_done=${RM75_BAD_PUSH_DONE:-False}"
  "+env.task_kwargs.reward_kwargs.bad_push_min_step=${RM75_BAD_PUSH_MIN_STEP:-6}"
  "+env.task_kwargs.reward_kwargs.bad_push_drift_thresh=${RM75_BAD_PUSH_DRIFT_THRESH:-0.08}"
  "+env.task_kwargs.reward_kwargs.bad_push_tilt_thresh=${RM75_BAD_PUSH_TILT_THRESH:-0.75}"
  "n_envs=${N_ENVS}"
  "n_eval_envs=${N_EVAL_ENVS}"
  "total_timesteps=${TOTAL_TIMESTEPS}"
  "eval_freq=${EVAL_FREQ}"
  "eval_n_episodes=${EVAL_N_EPISODES}"
  "save_freq=${SAVE_FREQ}"
  "restore_checkpoint_freq=${RESTORE_FREQ}"
  "hydra.run.dir=${RUN_DIR}"
  "wandb.project=${WANDB_PROJECT:-vividex_rm75_native}"
  "wandb.group=${WANDB_GROUP:-rm75_native_rfpo}"
  "wandb.sweep_name_prefix=${RUN_NAME}"
  "agent.params.actor_objective=gaussian_ppo"
  "agent.params.trust_region_mode=ppo"
  "agent.params.action_head_mode=${ACTION_HEAD_MODE:-flow_residual}"
  "agent.params.learning_rate=${LEARNING_RATE:-6e-6}"
  "agent.params.min_learning_rate=${MIN_LEARNING_RATE:-6e-6}"
  "agent.params.max_learning_rate=${MAX_LEARNING_RATE:-6e-6}"
  "agent.params.gaussian_action_std=${GAUSSIAN_ACTION_STD:-0.04}"
  "agent.params.gaussian_action_std_decay_steps=${STD_DECAY_STEPS:-400000}"
  "agent.params.gaussian_action_std_min_scale=${STD_MIN_SCALE:-0.50}"
  "agent.params.clip_range=${CLIP_RANGE:-0.16}"
  "agent.params.residual_action_scale=${RESIDUAL_ACTION_SCALE:-0.025}"
  "agent.params.rollout_deterministic=${ROLLOUT_DETERMINISTIC:-False}"
  "agent.params.eval_deterministic=${EVAL_DETERMINISTIC:-True}"
  "agent.params.n_steps=${N_STEPS}"
  "agent.params.batch_size=${BATCH_SIZE}"
  "agent.params.n_epochs=${N_EPOCHS}"
  "agent.params.n_samples_per_action=${N_SAMPLES}"
  "agent.params.sampling_steps=8"
  "agent.params.actor_hidden_dims=${ACTOR_DIMS}"
  "agent.params.critic_hidden_dims=${CRITIC_DIMS}"
  "agent.params.residual_head_hidden_dims=${RESIDUAL_DIMS}"
  "agent.params.direct_head_hidden_dims=${DIRECT_DIMS}"
  "agent.params.direct_action_scale=${DIRECT_ACTION_SCALE:-1.0}"
  "agent.params.activation=elu"
  "agent.params.max_grad_norm=4.0"
  "agent.params.actor_max_grad_norm=1.5"
  "agent.params.critic_max_grad_norm=4.0"
  "agent.params.train_actor_after_iterations=${TRAIN_ACTOR_AFTER_ITERATIONS:-0}"
  "agent.params.target_kl=${TARGET_KL:-null}"
  "agent.params.max_clip_fraction=${MAX_CLIP_FRACTION:-null}"
  "agent.params.initial_curriculum_stage=${INITIAL_CURRICULUM_STAGE:-0}"
  "agent.params.curriculum_stage1_metric=${CURRICULUM_STAGE1_METRIC:-null}"
  "agent.params.curriculum_stage1_threshold=${CURRICULUM_STAGE1_THRESHOLD:-0.0}"
  "agent.params.curriculum_stage2_metric=${CURRICULUM_STAGE2_METRIC:-null}"
  "agent.params.curriculum_stage2_threshold=${CURRICULUM_STAGE2_THRESHOLD:-0.0}"
  "agent.params.best_metric=${BEST_METRIC:-eval/mean_norm_success_10}"
  "agent.params.best_metric_mode=${BEST_METRIC_MODE:-max}"
  "agent.params.save_best_model=True"
  "agent.params.degrade_metric=${DEGRADE_METRIC:-eval/mean_norm_success_10}"
  "agent.params.degrade_threshold=${DEGRADE_THRESHOLD:-0.05}"
  "agent.params.rollback_to_best_on_degrade=${ROLLBACK_TO_BEST_ON_DEGRADE:-True}"
  "agent.params.rollback_patience=${ROLLBACK_PATIENCE:-1}"
  "agent.params.reset_optimizer_on_rollback=${RESET_OPTIMIZER_ON_ROLLBACK:-True}"
)

if [[ "${USE_BC_ANCHOR}" == "1" ]]; then
  CMD+=(
    "agent.params.bc_anchor_dataset=${BC_ANCHOR_DATASET}"
    "agent.params.bc_anchor_coef=${BC_ANCHOR_COEF:-0.60}"
    "agent.params.bc_anchor_min_coef=${BC_ANCHOR_MIN_COEF:-0.20}"
    "agent.params.bc_anchor_decay_steps=${BC_ANCHOR_DECAY_STEPS:-500000}"
    "agent.params.bc_anchor_batch_size=${BC_ANCHOR_BATCH_SIZE:-256}"
    "agent.params.action_anchor_coef=${ACTION_ANCHOR_COEF:-0.15}"
    "agent.params.action_anchor_min_coef=${ACTION_ANCHOR_MIN_COEF:-0.05}"
    "agent.params.action_anchor_decay_steps=${ACTION_ANCHOR_DECAY_STEPS:-500000}"
    "agent.params.action_anchor_batch_size=${ACTION_ANCHOR_BATCH_SIZE:-256}"
  )
fi

if [[ "${NO_BC_CHECKPOINT:-0}" != "1" && -n "${BC_CHECKPOINT:-}" ]]; then
  CMD+=("agent.params.bc_checkpoint=${BC_CHECKPOINT}")
fi

if [[ -n "${RESUME_MODEL:-}" ]]; then
  CMD+=("resume_model=${RESUME_MODEL}" "agent.params.resume_load_optimizer=${RESUME_LOAD_OPTIMIZER:-false}")
fi

printf '[native-rfpo] urdf=%s\n' "${NATIVE_URDF}"
printf '[native-rfpo] traj=%s\n' "${TRAJ_PATH}"
printf '[native-rfpo] object_scale=%s\n' "${OBJECT_SCALE}"
printf '[native-rfpo] base_offset=%s base_rpy=%s base_yaw=%s\n' "${RM75_BASE_OFFSET}" "${RM75_BASE_RPY}" "${RM75_BASE_YAW}"
if [[ "${USE_BC_ANCHOR}" == "1" ]]; then
  printf '[native-rfpo] bc_anchor=%s\n' "${BC_ANCHOR_DATASET}"
fi
if [[ "${NO_BC_CHECKPOINT:-0}" != "1" && -n "${BC_CHECKPOINT:-}" ]]; then
  printf '[native-rfpo] bc_checkpoint=%s\n' "${BC_CHECKPOINT}"
fi
printf '[native-rfpo] command: %q ' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
