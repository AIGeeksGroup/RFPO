#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${VIVIDEX_VENV:-${REPO}/.venv-vividex}"
PYTHON="${VENV}/bin/python"

SEQ="${SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"
STAMP="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-rm75_native_ppo_${STAMP}}"
RUN_ROOT="${VIVIDEX_LOCAL_RUN_ROOT:-${REPO}/.local_runs}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_NAME}}"

NATIVE_URDF="${NATIVE_URDF:-${REPO}/assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_nocyl.urdf}"
TRAJ_SUBDIR="${TRAJ_SUBDIR:-rm75_inspire_right_native12_clean_scale065_x090_y150_f080}"
TRAJ_PATH="${REPO}/norm_trajectories/${TRAJ_SUBDIR}/${SEQ}.npz"
RETARGET_MODE="${RETARGET_MODE:-reachable}"
RETARGET_PALM_OFFSET="${RETARGET_PALM_OFFSET:-0.090 -0.150 -0.002}"
OBJECT_SCALE="${OBJECT_SCALE:-${VIVIDEX_YCB_OBJECT_SCALE:-0.60}}"

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
  "${RETARGET_CMD[@]}"
fi

N_ENVS="${N_ENVS:-4}"
N_EVAL_ENVS="${N_EVAL_ENVS:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
EVAL_FREQ="${EVAL_FREQ:-50000}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-12}"
SAVE_FREQ="${SAVE_FREQ:-100000}"
RESTORE_FREQ="${RESTORE_FREQ:-100000}"
MULTI_PROC="${MULTI_PROC:-False}"
N_STEPS="${N_STEPS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-512}"
N_EPOCHS="${N_EPOCHS:-5}"
RESUME_MODEL="${RESUME_MODEL:-}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  N_ENVS="${SMOKE_N_ENVS:-1}"
  N_EVAL_ENVS="${SMOKE_N_EVAL_ENVS:-1}"
  TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS:-64}"
  EVAL_FREQ="${SMOKE_EVAL_FREQ:-128}"
  EVAL_N_EPISODES="${SMOKE_EVAL_N_EPISODES:-1}"
  SAVE_FREQ="${SMOKE_SAVE_FREQ:-128}"
  RESTORE_FREQ="${SMOKE_RESTORE_FREQ:-128}"
  MULTI_PROC=False
  N_STEPS="${SMOKE_N_STEPS:-8}"
  BATCH_SIZE="${SMOKE_BATCH_SIZE:-8}"
  N_EPOCHS="${SMOKE_N_EPOCHS:-1}"
fi

INFO_KEYS="[pregrasp_success,pregrasp_steps,imitate_steps,hand_jpos_err,obj_com_err,obj_rot_err,obj_lift,contact_count,stable_grasp_contact,thumb_contact,non_thumb_contact_count,palm_contact,pinky_contact,ignore_pinky_contact,stable_contact_hold_steps,rm75_grasp_score,rm75_precision_score,rm75_task_score,fingertip_obj_dist,min_fingertip_obj_dist,palm_obj_dist,hand_close_fraction,hand_thumb_close_fraction,hand_main_close_fraction,hand_pinky_close_fraction,reference_close_fraction,hand_reference_close_error,done_on_norm_success_10_active,contact_hold_steps,contact_success,lift_success_5cm,lift_success_target,norm_success_3,norm_success_10,object_xy_drift,object_xy_drift_x,object_xy_drift_y,object_speed,object_tilt_err,object_ang_speed,hand_mjpos_err,stage,control_error,obj_tgt_dist]"

CMD=(
  "${PYTHON}" tools/train.py
  agent=ppo
  "agent.multi_proc=${MULTI_PROC}"
  "env.name=${SEQ}"
  "env.robot_name=rm75_inspire_right"
  "env.norm_traj=True"
  "+env.task_kwargs.rm75_native_hand_control=True"
  "+env.task_kwargs.rm75_enforce_native_mimic_qpos=False"
  "+env.task_kwargs.rm75_pregrasp_success_mode=proximity"
  "+env.task_kwargs.rm75_native_apply_template_approach_pregrasp=True"
  "+env.task_kwargs.rm75_pregrasp_palm_dist_thresh=0.22"
  "+env.task_kwargs.rm75_pregrasp_min_finger_dist_thresh=0.10"
  "+env.task_kwargs.rm75_done_on_pregrasp_failure=False"
  "+env.task_kwargs.object_scale=${OBJECT_SCALE}"
  "+env.task_kwargs.rm75_default_approach_delta=[${RM75_DEFAULT_APPROACH_DX:--0.0204},${RM75_DEFAULT_APPROACH_DY:--0.1595},${RM75_DEFAULT_APPROACH_DZ:-0.1186}]"
  "+env.task_kwargs.rm75_override_template_approach_delta=True"
  "+env.task_kwargs.rm75_reset_settle_steps=80"
  "+env.task_kwargs.rm75_force_imitate_steps=260"
  "+env.task_kwargs.rm75_done_on_norm_success_10=True"
  "+env.task_kwargs.rm75_norm_success_lift_thresh=0.08"
  "+env.task_kwargs.rm75_norm_success_requires_contact=True"
  "+env.task_kwargs.rm75_success_min_non_thumb_contacts=2"
  "+env.task_kwargs.rm75_score_lift_target=0.08"
  "+env.task_kwargs.rm75_ignore_pinky_contact=True"
  "+env.task_kwargs.rm75_disable_pinky_action=True"
  "+env.task_kwargs.rm75_pinky_action_value=-0.5"
  "+env.task_kwargs.rm75_scripted_action_prior=False"
  "+env.task_kwargs.rm75_scripted_action_prior_blend=0.0"
  "env.info_keywords=${INFO_KEYS}"
  "+env.task_kwargs.reward_kwargs.pregrasp_success_thresh=0.085"
  "+env.task_kwargs.reward_kwargs.obj_com_done_thresh=0.35"
  "+env.task_kwargs.reward_kwargs.no_contact_grace_steps=260"
  "+env.task_kwargs.reward_kwargs.controller_penalty_scale=700.0"
  "+env.task_kwargs.reward_kwargs.action_penalty_scale=0.008"
  "+env.task_kwargs.reward_kwargs.reward_divisor=10.0"
  "+env.task_kwargs.reward_kwargs.finger_approach_reward_scale=2.0"
  "+env.task_kwargs.reward_kwargs.finger_approach_scale=22.0"
  "+env.task_kwargs.reward_kwargs.min_finger_approach_reward_scale=2.0"
  "+env.task_kwargs.reward_kwargs.min_finger_approach_scale=18.0"
  "+env.task_kwargs.reward_kwargs.palm_approach_reward_scale=0.0"
  "+env.task_kwargs.reward_kwargs.palm_approach_scale=7.0"
  "+env.task_kwargs.reward_kwargs.contact_reward_scale=0.0"
  "+env.task_kwargs.reward_kwargs.thumb_contact_reward_scale=6.0"
  "+env.task_kwargs.reward_kwargs.non_thumb_contact_reward_scale=6.0"
  "+env.task_kwargs.reward_kwargs.required_non_thumb_contacts=2"
  "+env.task_kwargs.reward_kwargs.object_reward_requires_contact=True"
  "+env.task_kwargs.reward_kwargs.object_reward_requires_stable_contact=True"
  "+env.task_kwargs.reward_kwargs.object_reward_contact_hold_steps=2"
  "env.task_kwargs.reward_kwargs.object_reward_scale=9.0"
  "env.task_kwargs.reward_kwargs.obj_err_scale=45.0"
  "+env.task_kwargs.reward_kwargs.obj_rot_term=0.1"
  "+env.task_kwargs.reward_kwargs.hand_mimic_reward_scale=1.2"
  "+env.task_kwargs.reward_kwargs.hand_mimic_scale=8.0"
  "+env.task_kwargs.reward_kwargs.hand_close_reward_scale=0.80"
  "+env.task_kwargs.reward_kwargs.hand_open_penalty_scale=2.20"
  "+env.task_kwargs.reward_kwargs.hand_close_target=0.80"
  "+env.task_kwargs.reward_kwargs.reference_close_reward_scale=0.25"
  "+env.task_kwargs.reward_kwargs.reference_close_penalty_scale=1.20"
  "+env.task_kwargs.reward_kwargs.reference_close_tolerance=0.04"
  "+env.task_kwargs.reward_kwargs.early_close_penalty_scale=0.15"
  "+env.task_kwargs.reward_kwargs.hand_synergy_reward_scale=0.40"
  "+env.task_kwargs.reward_kwargs.hand_synergy_penalty_scale=0.20"
  "+env.task_kwargs.reward_kwargs.hand_synergy_min_main_close=0.18"
  "+env.task_kwargs.reward_kwargs.stable_contact_bonus=18.0"
  "+env.task_kwargs.reward_kwargs.contact_hold_bonus_scale=0.0"
  "+env.task_kwargs.reward_kwargs.contact_hold_bonus_steps=6"
  "env.task_kwargs.reward_kwargs.lift_bonus_thresh=0.012"
  "env.task_kwargs.reward_kwargs.lift_bonus_mag=8.0"
  "+env.task_kwargs.reward_kwargs.lift_reward_scale=80.0"
  "+env.task_kwargs.reward_kwargs.lift_reward_cap=0.08"
  "+env.task_kwargs.reward_kwargs.object_xy_drift_free_thresh=0.018"
  "+env.task_kwargs.reward_kwargs.object_xy_drift_penalty_scale=90.0"
  "+env.task_kwargs.reward_kwargs.object_speed_penalty_scale=1.0"
  "+env.task_kwargs.reward_kwargs.object_tilt_penalty_scale=10.0"
  "+env.task_kwargs.reward_kwargs.object_ang_vel_penalty_scale=0.0"
  "+env.task_kwargs.reward_kwargs.unstable_push_penalty_scale=35.0"
  "+env.task_kwargs.reward_kwargs.bad_push_done=False"
  "n_envs=${N_ENVS}"
  "n_eval_envs=${N_EVAL_ENVS}"
  "total_timesteps=${TOTAL_TIMESTEPS}"
  "eval_freq=${EVAL_FREQ}"
  "eval_n_episodes=${EVAL_N_EPISODES}"
  "save_freq=${SAVE_FREQ}"
  "restore_checkpoint_freq=${RESTORE_FREQ}"
  "hydra.run.dir=${RUN_DIR}"
  "wandb.project=${WANDB_PROJECT:-vividex_rm75_native}"
  "wandb.group=${WANDB_GROUP:-rm75_native_ppo_baseline}"
  "wandb.sweep_name_prefix=${RUN_NAME}"
  "agent.params.learning_rate=${PPO_LR:-1e-5}"
  "agent.params.n_steps=${N_STEPS}"
  "agent.params.batch_size=${BATCH_SIZE}"
  "agent.params.n_epochs=${N_EPOCHS}"
  "agent.params.eval_deterministic=${EVAL_DETERMINISTIC:-False}"
)

if [[ -n "${RESUME_MODEL}" ]]; then
  CMD+=("resume_model=${RESUME_MODEL}")
fi

if [[ -n "${BC_ANCHOR_DATASET:-}" ]]; then
  CMD+=(
    "agent.params.bc_anchor_dataset=${BC_ANCHOR_DATASET}"
    "agent.params.bc_anchor_coef=${BC_ANCHOR_COEF:-1.0}"
    "agent.params.bc_anchor_min_coef=${BC_ANCHOR_MIN_COEF:-0.2}"
    "agent.params.bc_anchor_decay_steps=${BC_ANCHOR_DECAY_STEPS:-1000000}"
    "agent.params.bc_anchor_batch_size=${BC_ANCHOR_BATCH_SIZE:-512}"
    "agent.params.bc_anchor_updates=${BC_ANCHOR_UPDATES:-2}"
    "agent.params.bc_anchor_loss=${BC_ANCHOR_LOSS:-huber}"
    "agent.params.bc_anchor_huber_delta=${BC_ANCHOR_HUBER_DELTA:-0.05}"
    "agent.params.bc_anchor_grad_clip=${BC_ANCHOR_GRAD_CLIP:-1.0}"
  )
fi

printf '[rm75-native-ppo] run_dir=%s\n' "${RUN_DIR}"
printf '[rm75-native-ppo] command: %q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
