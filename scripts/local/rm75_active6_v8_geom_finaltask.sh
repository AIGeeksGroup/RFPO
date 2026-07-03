#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_RESUME="${REPO}/.local_runs/rm75_active6_v5_train_postcontactwrist_950k_20260625_211654/models/best.pt"
RESUME_MODEL="${RESUME_MODEL:-${DEFAULT_RESUME}}"

if [[ ! -f "${RESUME_MODEL}" ]]; then
  echo "Missing RESUME_MODEL: ${RESUME_MODEL}" >&2
  echo "Set RESUME_MODEL=/path/to/best.pt or run the v5 checkpoint first." >&2
  exit 1
fi

export VIVIDEX_HEADLESS_NO_RENDER="${VIVIDEX_HEADLESS_NO_RENDER:-1}"
export VIVIDEX_RM75_FINGER_FORCE_LIMIT="${VIVIDEX_RM75_FINGER_FORCE_LIMIT:-120}"
export VIVIDEX_RM75_FINGER_STIFFNESS="${VIVIDEX_RM75_FINGER_STIFFNESS:-500}"
export VIVIDEX_RM75_FINGER_DAMPING="${VIVIDEX_RM75_FINGER_DAMPING:-100}"
export VIVIDEX_RM75_STATIC_FRICTION="${VIVIDEX_RM75_STATIC_FRICTION:-6.0}"
export VIVIDEX_RM75_DYNAMIC_FRICTION="${VIVIDEX_RM75_DYNAMIC_FRICTION:-5.0}"
export VIVIDEX_RM75_RESTITUTION="${VIVIDEX_RM75_RESTITUTION:-0.0}"
export VIVIDEX_YCB_STATIC_FRICTION="${VIVIDEX_YCB_STATIC_FRICTION:-6.0}"
export VIVIDEX_YCB_DYNAMIC_FRICTION="${VIVIDEX_YCB_DYNAMIC_FRICTION:-5.0}"
export VIVIDEX_YCB_RESTITUTION="${VIVIDEX_YCB_RESTITUTION:-0.0}"

export RUN_NAME="${RUN_NAME:-rm75_active6_v8_geom_finaltask_$(date +%Y%m%d_%H%M%S)}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1100000}"
export EVAL_FREQ="${EVAL_FREQ:-25000}"
export EVAL_N_EPISODES="${EVAL_N_EPISODES:-20}"
export SAVE_FREQ="${SAVE_FREQ:-50000}"
export RESTORE_FREQ="${RESTORE_FREQ:-50000}"

OVERRIDES=(
  "resume_model=${RESUME_MODEL}"
  "agent.params.resume_load_optimizer=False"
  "env.task_kwargs.rm75_default_approach_delta=[0.02,-0.03,-0.01]"
  "env.task_kwargs.rm75_post_contact_wrist_bias=[-0.2,-0.45,0.0]"
  "env.task_kwargs.rm75_post_contact_wrist_min_hold_steps=1"
  "env.task_kwargs.rm75_post_contact_wrist_min_non_thumb_contacts=0"
  "env.task_kwargs.rm75_post_contact_wrist_requires_thumb=True"
  "env.task_kwargs.rm75_post_contact_lift_assist=True"
  "env.task_kwargs.rm75_post_contact_lift_min_hold_steps=1"
  "env.task_kwargs.rm75_post_contact_lift_min_non_thumb_contacts=0"
  "env.task_kwargs.rm75_post_contact_lift_requires_thumb=True"
  "env.task_kwargs.rm75_post_contact_max_xy_speed=0.006"
  "env.task_kwargs.rm75_post_contact_max_up_speed=0.045"
  "env.task_kwargs.rm75_post_contact_lift_bias=0.010"
  "env.task_kwargs.rm75_post_contact_max_angular_speed=0.24"
  "env.task_kwargs.rm75_hand_dynamic_close_bias=0.82"
  "env.task_kwargs.rm75_hand_dynamic_close_contact_lock_alpha=0.92"
  "env.task_kwargs.rm75_hand_dynamic_close_contact_boost=0.38"
  "env.task_kwargs.rm75_hand_precontact_close_cap=0.48"
  "env.task_kwargs.rm75_hand_precontact_thumb_cap=0.30"
  "env.task_kwargs.rm75_hand_precontact_pinky_cap=0.38"
  "env.task_kwargs.rm75_hand_close_bias_weights=[0.50,0.62,0.92,0.92,0.90,0.78]"
  "env.task_kwargs.reward_kwargs.no_contact_grace_steps=24"
  "env.task_kwargs.reward_kwargs.non_thumb_contact_reward_scale=2.8"
  "env.task_kwargs.reward_kwargs.stable_contact_bonus=16.0"
  "env.task_kwargs.reward_kwargs.contact_hold_bonus_scale=8.0"
  "env.task_kwargs.reward_kwargs.lift_bonus_thresh=0.004"
  "env.task_kwargs.reward_kwargs.lift_bonus_mag=9.0"
  "env.task_kwargs.reward_kwargs.lift_reward_scale=95.0"
  "env.task_kwargs.reward_kwargs.object_reward_contact_hold_steps=1"
  "env.task_kwargs.reward_kwargs.object_xy_drift_free_thresh=0.013"
  "env.task_kwargs.reward_kwargs.object_xy_drift_penalty_scale=24.0"
  "env.task_kwargs.reward_kwargs.object_tilt_free_thresh=0.22"
  "env.task_kwargs.reward_kwargs.object_tilt_penalty_scale=9.0"
  "env.task_kwargs.reward_kwargs.object_speed_penalty_scale=0.07"
  "env.task_kwargs.reward_kwargs.object_ang_vel_penalty_scale=0.045"
  "env.task_kwargs.reward_kwargs.unstable_push_penalty_scale=15.0"
  "env.task_kwargs.reward_kwargs.bad_push_done=False"
  "agent.params.rollback_to_best_on_degrade=False"
  "agent.params.best_metric=eval/mean_rm75_task_score"
  "agent.params.degrade_metric=eval/mean_rm75_task_score"
  "agent.params.degrade_threshold=8.0"
  "agent.params.eval_deterministic=True"
  "agent.params.rollout_action_noise_std=0.006"
  "agent.params.learning_rate=3.0e-6"
)

export EXTRA_OVERRIDES="${OVERRIDES[*]}"

cd "${REPO}"
exec bash scripts/local/rm75_active6_rfpo_train.sh
