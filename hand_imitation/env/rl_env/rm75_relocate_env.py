from __future__ import annotations

import os

import numpy as np

from hand_imitation.env.rl_env.reference_utils import (
    RM75_CLOSED_HAND_QPOS,
    RM75_ACTIVE_HAND_QPOS_INDICES,
    RM75_PREGRASP_HAND_QPOS,
    compute_rm75_grasp_scores,
    compute_rm75_task_score,
    rm75_active_hand_close_state,
    rm75_native_active_qpos_from_full,
    should_terminate_for_contact_loss,
)
from hand_imitation.env.rl_env.relocate_env import AllegroRelocateRLEnv, rotation_distance
from hand_imitation.env.rl_env.rm75_reward import (
    RM75RewardConfig,
    compute_rm75_reward,
    has_stable_rm75_contact,
)


class RM75RelocateRLEnv(AllegroRelocateRLEnv):
    """RM75/RH56-specific relocation RL environment.

    The original Allegro/UR5 environment stays intact. This subclass keeps the
    same reset, observation, curriculum, and Cartesian action logic, but uses a
    reward and termination rule tuned for the RM75 arm plus coupled RH56 hand.
    """

    def __init__(self, *args, task_kwargs=None, robot_name="rm75_inspire_right", **kwargs):
        task_kwargs = task_kwargs or {}
        self.rm75_reward_cfg = RM75RewardConfig.from_kwargs(task_kwargs.get("reward_kwargs", {}))
        self.robot_name = robot_name
        self.stable_grasp_contact = False
        self.contact_count = 0.0
        env_native = os.environ.get("VIVIDEX_RM75_NATIVE_HAND_CONTROL", "0") == "1"
        self.rm75_native_hand_control = bool(task_kwargs.get("rm75_native_hand_control", env_native))
        env_enforce_mimic = os.environ.get("VIVIDEX_RM75_ENFORCE_NATIVE_MIMIC", "0") == "1"
        self.rm75_enforce_native_mimic_qpos = bool(
            task_kwargs.get("rm75_enforce_native_mimic_qpos", env_enforce_mimic)
        )
        self.rm75_native_apply_template_approach_pregrasp = bool(
            task_kwargs.get("rm75_native_apply_template_approach_pregrasp", False)
        )
        default_pregrasp_mode = "proximity" if self.rm75_native_hand_control else "reference"
        self.rm75_pregrasp_success_mode = str(
            task_kwargs.get("rm75_pregrasp_success_mode", default_pregrasp_mode)
        )
        self.rm75_pregrasp_palm_dist_thresh = float(task_kwargs.get("rm75_pregrasp_palm_dist_thresh", 0.16))
        self.rm75_pregrasp_min_finger_dist_thresh = float(
            task_kwargs.get("rm75_pregrasp_min_finger_dist_thresh", 0.075)
        )
        self.rm75_done_on_pregrasp_failure = bool(task_kwargs.get("rm75_done_on_pregrasp_failure", True))
        self.rm75_robot_base_offset = np.asarray(
            task_kwargs.get("rm75_robot_base_offset", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        if self.rm75_robot_base_offset.shape != (3,):
            raise ValueError("rm75_robot_base_offset must contain 3 values")
        self.rm75_robot_base_rpy = np.asarray(
            task_kwargs.get("rm75_robot_base_rpy", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        if self.rm75_robot_base_rpy.shape != (3,):
            raise ValueError("rm75_robot_base_rpy must contain 3 values")
        self.rm75_robot_base_rpy[2] += float(task_kwargs.get("rm75_robot_base_yaw", 0.0))
        self.rm75_arm_action_scale = float(task_kwargs.get("rm75_arm_action_scale", 1.0))
        self.rm75_hand_action_scale = float(task_kwargs.get("rm75_hand_action_scale", 1.0))
        self.rm75_pregrasp_safe_action = bool(task_kwargs.get("rm75_pregrasp_safe_action", False))
        self.rm75_pregrasp_safe_margin_steps = int(task_kwargs.get("rm75_pregrasp_safe_margin_steps", 0))
        self.rm75_pregrasp_arm_action_scale = float(
            task_kwargs.get("rm75_pregrasp_arm_action_scale", self.rm75_arm_action_scale)
        )
        self.rm75_pregrasp_hand_action_scale = float(
            task_kwargs.get("rm75_pregrasp_hand_action_scale", self.rm75_hand_action_scale)
        )
        self.rm75_pregrasp_hand_close_bias = float(task_kwargs.get("rm75_pregrasp_hand_close_bias", 0.0))
        self.rm75_pregrasp_zero_hand_action = bool(task_kwargs.get("rm75_pregrasp_zero_hand_action", False))
        self.rm75_post_pregrasp_arm_ramp_steps = int(task_kwargs.get("rm75_post_pregrasp_arm_ramp_steps", 0))
        self.rm75_post_pregrasp_arm_scale_start = float(
            task_kwargs.get("rm75_post_pregrasp_arm_scale_start", self.rm75_arm_action_scale)
        )
        self.rm75_post_pregrasp_arm_scale_end = float(
            task_kwargs.get("rm75_post_pregrasp_arm_scale_end", self.rm75_arm_action_scale)
        )
        self.rm75_post_pregrasp_hold_until_stable = bool(task_kwargs.get("rm75_post_pregrasp_hold_until_stable", False))
        self.rm75_post_pregrasp_hold_arm_scale = float(task_kwargs.get("rm75_post_pregrasp_hold_arm_scale", 0.02))
        self.rm75_post_pregrasp_required_stable_hold_steps = int(
            task_kwargs.get("rm75_post_pregrasp_required_stable_hold_steps", 2)
        )
        self.rm75_post_pregrasp_project_to_object = bool(
            task_kwargs.get("rm75_post_pregrasp_project_to_object", False)
        )
        self.rm75_post_pregrasp_project_until_stable = bool(
            task_kwargs.get("rm75_post_pregrasp_project_until_stable", True)
        )
        self.rm75_post_pregrasp_project_tangent_scale = float(
            task_kwargs.get("rm75_post_pregrasp_project_tangent_scale", 0.18)
        )
        self.rm75_post_pregrasp_project_max_approach_speed = float(
            task_kwargs.get("rm75_post_pregrasp_project_max_approach_speed", 0.055)
        )
        self.rm75_post_pregrasp_project_max_retreat_speed = float(
            task_kwargs.get("rm75_post_pregrasp_project_max_retreat_speed", 0.012)
        )
        self.rm75_default_approach_delta = np.asarray(
            task_kwargs.get("rm75_default_approach_delta", [0.0, -0.05, 0.0]),
            dtype=np.float32,
        )
        if self.rm75_default_approach_delta.shape != (3,):
            raise ValueError("rm75_default_approach_delta must contain 3 values")
        self.rm75_override_template_approach_delta = bool(
            task_kwargs.get("rm75_override_template_approach_delta", False)
        )
        self.rm75_follow_reference_approach_delta = bool(
            task_kwargs.get("rm75_follow_reference_approach_delta", False)
        )
        self.rm75_min_imitate_steps = int(task_kwargs.get("rm75_min_imitate_steps", 0))
        self.rm75_force_imitate_steps = int(task_kwargs.get("rm75_force_imitate_steps", 0))
        self.rm75_reset_settle_steps = int(task_kwargs.get("rm75_reset_settle_steps", 0))
        self.rm75_post_stable_lift_only = bool(task_kwargs.get("rm75_post_stable_lift_only", False))
        self.rm75_post_stable_max_xy_speed = float(task_kwargs.get("rm75_post_stable_max_xy_speed", 0.012))
        self.rm75_post_stable_max_down_speed = float(task_kwargs.get("rm75_post_stable_max_down_speed", 0.002))
        self.rm75_post_stable_max_up_speed = float(task_kwargs.get("rm75_post_stable_max_up_speed", 0.035))
        self.rm75_post_stable_lift_bias = float(task_kwargs.get("rm75_post_stable_lift_bias", 0.0))
        self.rm75_post_stable_max_angular_speed = float(
            task_kwargs.get("rm75_post_stable_max_angular_speed", 0.20)
        )
        self.rm75_post_stable_xy_drift_correction_gain = float(
            task_kwargs.get("rm75_post_stable_xy_drift_correction_gain", 0.0)
        )
        self.rm75_post_stable_xy_drift_correction_max_speed = float(
            task_kwargs.get("rm75_post_stable_xy_drift_correction_max_speed", 0.0)
        )
        self.rm75_post_lift_latch_steps = int(task_kwargs.get("rm75_post_lift_latch_steps", 0))
        self.rm75_post_lift_latch_hand_alpha = float(task_kwargs.get("rm75_post_lift_latch_hand_alpha", 0.0))
        self.rm75_done_on_norm_success_10 = bool(task_kwargs.get("rm75_done_on_norm_success_10", False))
        self.rm75_norm_success_lift_thresh = float(task_kwargs.get("rm75_norm_success_lift_thresh", 0.05))
        self.rm75_norm_success_requires_contact = bool(task_kwargs.get("rm75_norm_success_requires_contact", True))
        self.rm75_success_min_non_thumb_contacts = int(
            task_kwargs.get("rm75_success_min_non_thumb_contacts", 1)
        )
        self.rm75_score_lift_target = float(task_kwargs.get("rm75_score_lift_target", 0.08))
        self.rm75_post_contact_lift_assist = bool(task_kwargs.get("rm75_post_contact_lift_assist", False))
        self.rm75_post_contact_lift_min_hold_steps = int(
            task_kwargs.get("rm75_post_contact_lift_min_hold_steps", 1)
        )
        self.rm75_post_contact_lift_min_non_thumb_contacts = int(
            task_kwargs.get("rm75_post_contact_lift_min_non_thumb_contacts", 0)
        )
        self.rm75_post_contact_lift_requires_thumb = bool(
            task_kwargs.get("rm75_post_contact_lift_requires_thumb", True)
        )
        self.rm75_post_contact_max_xy_speed = float(task_kwargs.get("rm75_post_contact_max_xy_speed", 0.014))
        self.rm75_post_contact_max_down_speed = float(task_kwargs.get("rm75_post_contact_max_down_speed", 0.002))
        self.rm75_post_contact_max_up_speed = float(task_kwargs.get("rm75_post_contact_max_up_speed", 0.055))
        self.rm75_post_contact_lift_bias = float(task_kwargs.get("rm75_post_contact_lift_bias", 0.0))
        self.rm75_post_contact_max_angular_speed = float(
            task_kwargs.get("rm75_post_contact_max_angular_speed", 0.16)
        )
        self.rm75_post_contact_xy_drift_correction_gain = float(
            task_kwargs.get("rm75_post_contact_xy_drift_correction_gain", 0.0)
        )
        self.rm75_post_contact_xy_drift_correction_max_speed = float(
            task_kwargs.get("rm75_post_contact_xy_drift_correction_max_speed", 0.0)
        )
        self.rm75_scripted_lift_prior = bool(task_kwargs.get("rm75_scripted_lift_prior", False))
        self.rm75_scripted_lift_prior_start_step = int(task_kwargs.get("rm75_scripted_lift_prior_start_step", 0))
        self.rm75_scripted_lift_prior_approach_steps = int(
            task_kwargs.get("rm75_scripted_lift_prior_approach_steps", 0)
        )
        self.rm75_scripted_lift_prior_close_steps = int(task_kwargs.get("rm75_scripted_lift_prior_close_steps", 18))
        self.rm75_scripted_lift_prior_hold_steps = int(task_kwargs.get("rm75_scripted_lift_prior_hold_steps", 10))
        self.rm75_scripted_lift_prior_lift_steps = int(task_kwargs.get("rm75_scripted_lift_prior_lift_steps", 45))
        self.rm75_scripted_lift_prior_lift_height = float(
            task_kwargs.get("rm75_scripted_lift_prior_lift_height", 0.10)
        )
        self.rm75_scripted_lift_prior_pos_gain = float(task_kwargs.get("rm75_scripted_lift_prior_pos_gain", 0.04))
        self.rm75_scripted_lift_prior_blend = float(task_kwargs.get("rm75_scripted_lift_prior_blend", 1.0))
        self.rm75_scripted_lift_prior_track_mode = str(
            task_kwargs.get("rm75_scripted_lift_prior_track_mode", "initial")
        )
        self.rm75_scripted_lift_prior_zero_angular = bool(
            task_kwargs.get("rm75_scripted_lift_prior_zero_angular", True)
        )
        self.rm75_scripted_hand_prior = bool(task_kwargs.get("rm75_scripted_hand_prior", False))
        self.rm75_scripted_hand_prior_blend = float(task_kwargs.get("rm75_scripted_hand_prior_blend", 1.0))
        self.rm75_scripted_hand_prior_approach_steps = int(
            task_kwargs.get("rm75_scripted_hand_prior_approach_steps", self.rm75_scripted_lift_prior_approach_steps)
        )
        self.rm75_scripted_hand_prior_close_steps = int(
            task_kwargs.get("rm75_scripted_hand_prior_close_steps", self.rm75_scripted_lift_prior_close_steps)
        )
        self.rm75_scripted_hand_prior_thumb_yaw = float(task_kwargs.get("rm75_scripted_hand_prior_thumb_yaw", 0.90))
        self.rm75_scripted_hand_prior_thumb_pitch = float(task_kwargs.get("rm75_scripted_hand_prior_thumb_pitch", 0.85))
        self.rm75_scripted_hand_prior_finger_close = float(task_kwargs.get("rm75_scripted_hand_prior_finger_close", 0.75))
        self.rm75_scripted_hand_prior_pinky_close = float(task_kwargs.get("rm75_scripted_hand_prior_pinky_close", 0.75))
        self.rm75_scripted_action_prior = bool(task_kwargs.get("rm75_scripted_action_prior", False))
        self.rm75_scripted_action_prior_blend = float(task_kwargs.get("rm75_scripted_action_prior_blend", 1.0))
        self.rm75_scripted_action_prior_approach_steps = int(
            task_kwargs.get("rm75_scripted_action_prior_approach_steps", 14)
        )
        self.rm75_scripted_action_prior_approach_z_power = float(
            task_kwargs.get("rm75_scripted_action_prior_approach_z_power", 1.0)
        )
        self.rm75_scripted_action_prior_close_steps = int(
            task_kwargs.get("rm75_scripted_action_prior_close_steps", 30)
        )
        self.rm75_scripted_action_prior_hold_steps = int(
            task_kwargs.get("rm75_scripted_action_prior_hold_steps", 36)
        )
        self.rm75_scripted_action_prior_lift_steps = int(
            task_kwargs.get("rm75_scripted_action_prior_lift_steps", 140)
        )
        self.rm75_scripted_action_prior_lift_height = float(
            task_kwargs.get("rm75_scripted_action_prior_lift_height", 0.18)
        )
        self.rm75_scripted_action_prior_pos_gain = float(task_kwargs.get("rm75_scripted_action_prior_pos_gain", 0.04))
        self.rm75_scripted_action_prior_mode = str(task_kwargs.get("rm75_scripted_action_prior_mode", "lift"))
        self.rm75_scripted_action_prior_target_max_xy_step = float(
            task_kwargs.get("rm75_scripted_action_prior_target_max_xy_step", 0.006)
        )
        self.rm75_scripted_action_prior_target_max_z_step = float(
            task_kwargs.get("rm75_scripted_action_prior_target_max_z_step", 0.006)
        )
        self.rm75_scripted_action_prior_target_gain = float(
            task_kwargs.get("rm75_scripted_action_prior_target_gain", 0.75)
        )
        self.rm75_scripted_action_prior_target_z_deadband = float(
            task_kwargs.get("rm75_scripted_action_prior_target_z_deadband", 0.002)
        )
        self.rm75_scripted_action_prior_target_pre_lift_height = float(
            task_kwargs.get("rm75_scripted_action_prior_target_pre_lift_height", 0.0)
        )
        self.rm75_scripted_action_prior_target_require_stable = bool(
            task_kwargs.get("rm75_scripted_action_prior_target_require_stable", False)
        )
        self.rm75_scripted_action_prior_thumb_yaw = float(task_kwargs.get("rm75_scripted_action_prior_thumb_yaw", 0.90))
        self.rm75_scripted_action_prior_thumb_pitch = float(task_kwargs.get("rm75_scripted_action_prior_thumb_pitch", 0.85))
        self.rm75_scripted_action_prior_finger_close = float(task_kwargs.get("rm75_scripted_action_prior_finger_close", 0.75))
        self.rm75_scripted_action_prior_pinky_close = float(task_kwargs.get("rm75_scripted_action_prior_pinky_close", 0.75))
        self.rm75_scripted_action_prior_grasp_offset = np.asarray(
            task_kwargs.get("rm75_scripted_action_prior_grasp_offset", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        if self.rm75_scripted_action_prior_grasp_offset.shape != (3,):
            raise ValueError("rm75_scripted_action_prior_grasp_offset must contain 3 values")
        self.rm75_scripted_action_prior_offset_pregrasp = bool(
            task_kwargs.get("rm75_scripted_action_prior_offset_pregrasp", False)
        )
        self.rm75_scripted_action_prior_thumb_delay_steps = int(
            task_kwargs.get("rm75_scripted_action_prior_thumb_delay_steps", 0)
        )
        self.rm75_scripted_action_prior_thumb_close_steps = int(
            task_kwargs.get("rm75_scripted_action_prior_thumb_close_steps", 0)
        )
        self.rm75_post_contact_wrist_bias = np.asarray(
            task_kwargs.get("rm75_post_contact_wrist_bias", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        if self.rm75_post_contact_wrist_bias.shape != (3,):
            raise ValueError("rm75_post_contact_wrist_bias must contain 3 values")
        self.rm75_post_contact_wrist_min_hold_steps = int(
            task_kwargs.get("rm75_post_contact_wrist_min_hold_steps", 1)
        )
        self.rm75_post_contact_wrist_min_non_thumb_contacts = int(
            task_kwargs.get("rm75_post_contact_wrist_min_non_thumb_contacts", 0)
        )
        self.rm75_post_contact_wrist_requires_thumb = bool(
            task_kwargs.get("rm75_post_contact_wrist_requires_thumb", True)
        )
        self.rm75_wrist_angular_bias = np.asarray(
            task_kwargs.get("rm75_wrist_angular_bias", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        if self.rm75_wrist_angular_bias.shape != (3,):
            raise ValueError("rm75_wrist_angular_bias must contain 3 values")
        self.rm75_wrist_angular_bias_start_step = int(task_kwargs.get("rm75_wrist_angular_bias_start_step", 0))
        self.rm75_wrist_angular_bias_end_step = int(task_kwargs.get("rm75_wrist_angular_bias_end_step", 1000000))
        thumb_action_cap = task_kwargs.get("rm75_thumb_action_cap_until_non_thumb", None)
        self.rm75_thumb_action_cap_until_non_thumb = (
            None if thumb_action_cap is None else float(thumb_action_cap)
        )
        self.rm75_thumb_action_cap_release_non_thumb_contacts = int(
            task_kwargs.get("rm75_thumb_action_cap_release_non_thumb_contacts", 2)
        )
        self.rm75_thumb_action_cap_release_contact_hold_steps = int(
            task_kwargs.get("rm75_thumb_action_cap_release_contact_hold_steps", 1)
        )
        self.rm75_ignore_pinky_contact = bool(task_kwargs.get("rm75_ignore_pinky_contact", False))
        self.rm75_disable_pinky_action = bool(task_kwargs.get("rm75_disable_pinky_action", False))
        self.rm75_pinky_action_value = float(task_kwargs.get("rm75_pinky_action_value", 0.0))
        self.rm75_hand_close_bias = float(task_kwargs.get("rm75_hand_close_bias", 0.0))
        self.rm75_hand_phase_close_bias = float(task_kwargs.get("rm75_hand_phase_close_bias", 0.0))
        self.rm75_hand_phase_close_start_step = int(task_kwargs.get("rm75_hand_phase_close_start_step", 0))
        self.rm75_hand_phase_close_ramp_steps = int(task_kwargs.get("rm75_hand_phase_close_ramp_steps", 1))
        self.rm75_hand_close_palm_dist = float(task_kwargs.get("rm75_hand_close_palm_dist", 0.22))
        self.rm75_hand_close_min_finger_dist = float(task_kwargs.get("rm75_hand_close_min_finger_dist", 0.12))
        self.rm75_hand_dynamic_close_bias = float(task_kwargs.get("rm75_hand_dynamic_close_bias", 0.0))
        self.rm75_hand_dynamic_close_start = float(task_kwargs.get("rm75_hand_dynamic_close_start", 0.22))
        self.rm75_hand_dynamic_close_full = float(task_kwargs.get("rm75_hand_dynamic_close_full", 0.08))
        self.rm75_hand_dynamic_close_requires_contact = bool(task_kwargs.get("rm75_hand_dynamic_close_requires_contact", False))
        self.rm75_hand_dynamic_close_contact_boost = float(task_kwargs.get("rm75_hand_dynamic_close_contact_boost", 0.0))
        self.rm75_hand_dynamic_close_non_thumb_thumb_alpha = float(
            task_kwargs.get("rm75_hand_dynamic_close_non_thumb_thumb_alpha", 0.0)
        )
        self.rm75_hand_precontact_close_cap = float(task_kwargs.get("rm75_hand_precontact_close_cap", 1.0))
        self.rm75_hand_precontact_thumb_cap = float(task_kwargs.get("rm75_hand_precontact_thumb_cap", self.rm75_hand_precontact_close_cap))
        self.rm75_hand_precontact_pinky_cap = float(task_kwargs.get("rm75_hand_precontact_pinky_cap", self.rm75_hand_precontact_close_cap))
        self.rm75_hand_dynamic_close_thumb_delay = float(task_kwargs.get("rm75_hand_dynamic_close_thumb_delay", 0.18))
        self.rm75_hand_dynamic_close_pinky_delay = float(task_kwargs.get("rm75_hand_dynamic_close_pinky_delay", 0.10))
        self.rm75_hand_dynamic_close_finger_mode = str(task_kwargs.get("rm75_hand_dynamic_close_finger_mode", "mean"))
        self.rm75_hand_dynamic_close_gate_mode = str(task_kwargs.get("rm75_hand_dynamic_close_gate_mode", "blend"))
        self.rm75_hand_dynamic_close_palm_weight = float(task_kwargs.get("rm75_hand_dynamic_close_palm_weight", 0.55))
        self.rm75_hand_dynamic_close_finger_weight = float(task_kwargs.get("rm75_hand_dynamic_close_finger_weight", 0.45))
        self.rm75_hand_dynamic_close_warmup_steps = int(task_kwargs.get("rm75_hand_dynamic_close_warmup_steps", 0))
        self.rm75_hand_dynamic_close_ramp_steps = int(task_kwargs.get("rm75_hand_dynamic_close_ramp_steps", 1))
        self.rm75_hand_dynamic_close_alpha_smooth = float(task_kwargs.get("rm75_hand_dynamic_close_alpha_smooth", 0.0))
        self.rm75_hand_dynamic_close_drift_gate = float(task_kwargs.get("rm75_hand_dynamic_close_drift_gate", np.inf))
        self.rm75_hand_dynamic_close_contact_hold_steps = int(task_kwargs.get("rm75_hand_dynamic_close_contact_hold_steps", 1))
        self.rm75_hand_dynamic_close_contact_lock_alpha = float(task_kwargs.get("rm75_hand_dynamic_close_contact_lock_alpha", 0.0))
        self.rm75_hand_dynamic_close_phase_weight = float(task_kwargs.get("rm75_hand_dynamic_close_phase_weight", 0.0))
        self.rm75_hand_dynamic_close_phase_start = int(task_kwargs.get("rm75_hand_dynamic_close_phase_start", 0))
        self.rm75_hand_dynamic_close_phase_ramp = int(task_kwargs.get("rm75_hand_dynamic_close_phase_ramp", 24))
        self.rm75_hand_dynamic_close_phase_near_gate = float(task_kwargs.get("rm75_hand_dynamic_close_phase_near_gate", 0.28))
        self.rm75_hand_dynamic_close_group_mode = str(task_kwargs.get("rm75_hand_dynamic_close_group_mode", "delay"))
        self.rm75_hand_dynamic_close_thumb_ratio = float(task_kwargs.get("rm75_hand_dynamic_close_thumb_ratio", 0.72))
        self.rm75_hand_dynamic_close_pinky_ratio = float(task_kwargs.get("rm75_hand_dynamic_close_pinky_ratio", 0.78))
        bias_weights = task_kwargs.get("rm75_hand_close_bias_weights", [0.55, 0.65, 1.0, 1.0, 0.9, 0.75])
        self.rm75_hand_close_bias_weights = np.asarray(bias_weights, dtype=np.float32)
        if self.rm75_hand_close_bias_weights.shape != (6,):
            raise ValueError("rm75_hand_close_bias_weights must contain 6 values")
        self.rm75_contact_hold_steps = 0
        self.rm75_stable_contact_hold_steps = 0
        self.rm75_last_dynamic_close_alpha = 0.0
        self.rm75_last_dynamic_close_group_alpha = np.zeros(6, dtype=np.float32)
        self.rm75_thumb_contact = False
        self.rm75_non_thumb_contact_count = 0
        self.rm75_palm_contact = False
        self.rm75_pregrasp_safe_active = False
        self.rm75_last_effective_arm_scale = self.rm75_arm_action_scale
        self.rm75_last_effective_hand_scale = self.rm75_hand_action_scale
        self.rm75_last_project_to_object_active = False
        self.rm75_last_post_stable_lift_only_active = False
        self.rm75_last_post_contact_lift_assist_active = False
        self.rm75_last_post_contact_wrist_bias_active = False
        self.rm75_last_scripted_lift_prior_active = False
        self.rm75_lift_latch_steps_left = 0
        self.rm75_target_track_latched = False
        self.rm75_scripted_action_prior_target_latched = False
        self.rm75_done_on_norm_success_10_active = False
        super().__init__(*args, task_kwargs=task_kwargs, robot_name=robot_name, **kwargs)
        self._seed_rm75_template_approach_reference()

    def _seed_rm75_template_approach_reference(self) -> None:
        if self.robot_name != "rm75_inspire_right":
            return
        if bool(getattr(self, "rm75_native_hand_control", False)) and not bool(
            getattr(self, "rm75_native_apply_template_approach_pregrasp", False)
        ):
            return
        motion = getattr(self, "_reference_motion", None)
        if motion is None or "robot_jpos" not in motion or "robot_pregrasp_jpos" not in motion:
            return
        if "rm75_template_approach_delta" in motion and not bool(
            getattr(self, "rm75_override_template_approach_delta", False)
        ):
            approach_delta = np.asarray(motion["rm75_template_approach_delta"], dtype=np.float32)
        else:
            approach_delta = np.asarray(
                getattr(self, "rm75_default_approach_delta", [0.0, -0.05, 0.0]),
                dtype=np.float32,
            )
            motion["rm75_template_approach_delta"] = approach_delta.copy()
        if approach_delta.shape != (3,):
            return
        grasp_palm = np.asarray(motion["robot_jpos"][0, 0], dtype=np.float32)
        motion["robot_pregrasp_jpos"][-1, 0] = grasp_palm + approach_delta
        if hasattr(self, "cur_reference_motion") and isinstance(self.cur_reference_motion, dict):
            self.cur_reference_motion["rm75_template_approach_delta"] = approach_delta.copy()
            if "robot_pregrasp_jpos" in self.cur_reference_motion and "robot_jpos" in self.cur_reference_motion:
                cur_grasp_palm = np.asarray(self.cur_reference_motion["robot_jpos"][0, 0], dtype=np.float32)
                self.cur_reference_motion["robot_pregrasp_jpos"][-1, 0] = cur_grasp_palm + approach_delta

    def reset_internal(self):
        self.rm75_contact_hold_steps = 0
        self.rm75_stable_contact_hold_steps = 0
        self.rm75_last_dynamic_close_alpha = 0.0
        self.rm75_last_dynamic_close_group_alpha = np.zeros(6, dtype=np.float32)
        self.rm75_thumb_contact = False
        self.rm75_non_thumb_contact_count = 0
        self.rm75_palm_contact = False
        self.rm75_pregrasp_safe_active = False
        self.rm75_last_effective_arm_scale = self.rm75_arm_action_scale
        self.rm75_last_effective_hand_scale = self.rm75_hand_action_scale
        self.rm75_last_project_to_object_active = False
        self.rm75_last_post_stable_lift_only_active = False
        self.rm75_last_post_contact_lift_assist_active = False
        self.rm75_last_post_contact_wrist_bias_active = False
        self.rm75_last_scripted_lift_prior_active = False
        self.rm75_lift_latch_steps_left = 0
        self.rm75_target_track_latched = False
        self.rm75_scripted_action_prior_target_latched = False
        self.rm75_done_on_norm_success_10_active = False
        super().reset_internal()

    def _post_reset_settle(self) -> None:
        settle_steps = max(int(getattr(self, "rm75_reset_settle_steps", 0)), 0)
        if settle_steps > 0:
            target_qpos = np.asarray(self.robot.get_qpos(), dtype=np.float32).copy()
            self.robot.set_drive_target(target_qpos)
            self.robot.set_drive_velocity_target(np.zeros_like(target_qpos))
            for _ in range(settle_steps):
                self._zero_manipulated_object_velocity()
                self.robot.set_qf(self.robot.compute_passive_force(external=False, coriolis_and_centrifugal=False))
                self.scene.step()

    def _safe_reference_index(self):
        return max(0, min(int(self.current_step - self.pregrasp_steps), len(self.cur_reference_motion["object_translation"]) - 1))

    def _pregrasp_reference_hand_pos(self):
        if "robot_pregrasp_jpos" in self.cur_reference_motion:
            return self._reference_hand_jpos(self.cur_reference_motion["robot_pregrasp_jpos"][-1])
        return self._reference_hand_jpos(self.cur_reference_motion["robot_jpos"][0])

    def _current_fingertip_pos(self):
        robot_hand_pos = np.zeros([len(self.finger_tip_links), 3], dtype=np.float32)
        for i, link in enumerate(self.finger_tip_links):
            robot_hand_pos[i] = link.get_pose().p
        return robot_hand_pos

    def _reference_active_hand_qpos(self) -> np.ndarray:
        motion = getattr(self, "cur_reference_motion", {})
        robot_qpos = motion.get("robot_qpos") if isinstance(motion, dict) else None
        idx = self._safe_reference_index()
        if robot_qpos is not None:
            robot_qpos = np.asarray(robot_qpos, dtype=np.float32)
            if robot_qpos.ndim == 2 and robot_qpos.shape[0] > 0:
                qpos = robot_qpos[min(idx, robot_qpos.shape[0] - 1)]
                hand_qpos = qpos[int(getattr(self, "arm_dof", 7)):]
                if hand_qpos.shape[0] == 6:
                    return hand_qpos.copy()
                if hand_qpos.shape[0] == 12:
                    if bool(getattr(self, "rm75_native_hand_control", False)):
                        return hand_qpos.copy()
                    return hand_qpos[RM75_ACTIVE_HAND_QPOS_INDICES].copy()
        if len(motion.get("object_translation", [])) <= 1 if isinstance(motion, dict) else True:
            alpha = 0.0
        else:
            alpha = float(np.clip(idx / max(len(motion["object_translation"]) - 1, 1), 0.0, 1.0))
        return (1.0 - alpha) * RM75_PREGRASP_HAND_QPOS + alpha * RM75_CLOSED_HAND_QPOS

    def _reference_close_fraction(self) -> float:
        active_qpos = self._reference_active_hand_qpos()
        if bool(getattr(self, "rm75_native_hand_control", False)) and active_qpos.shape[0] != 6:
            qlimits = np.asarray(self.robot.get_qlimits()[self.arm_dof:], dtype=np.float32)
            close_fraction, thumb_close, main_close, pinky_close = self._native_close_state_from_qpos(
                active_qpos,
                qlimits,
            )
        else:
            close_fraction, thumb_close, main_close, pinky_close = rm75_active_hand_close_state(active_qpos)
        self.reference_close_fraction = close_fraction
        self.reference_thumb_close_fraction = thumb_close
        self.reference_main_close_fraction = main_close
        self.reference_pinky_close_fraction = pinky_close
        return close_fraction

    def _native_close_state_from_qpos(self, qpos: np.ndarray, qlimits: np.ndarray) -> tuple[float, float, float, float]:
        qpos = np.asarray(qpos, dtype=np.float32)
        qlimits = np.asarray(qlimits, dtype=np.float32)
        if qpos.shape[0] == 12 and qlimits.shape[0] == 12:
            return rm75_active_hand_close_state(rm75_native_active_qpos_from_full(qpos))
        if qpos.shape[0] != qlimits.shape[0] or qpos.shape[0] == 0:
            return 0.0, 0.0, 0.0, 0.0
        denom = np.maximum(qlimits[:, 1] - qlimits[:, 0], 1e-6)
        close = np.clip((qpos - qlimits[:, 0]) / denom, 0.0, 1.0)
        thumb_count = min(4, close.shape[0])
        thumb_close = float(np.mean(close[:thumb_count])) if thumb_count > 0 else 0.0
        main_close = float(np.mean(close[thumb_count:])) if close.shape[0] > thumb_count else 0.0
        pinky_close = float(close[-1])
        close_fraction = float(np.mean(close))
        return close_fraction, thumb_close, main_close, pinky_close

    def rm75_dynamic_hand_close_bias(self) -> np.ndarray:
        if self.robot_name != "rm75_inspire_right" or self.current_step <= self.pregrasp_steps:
            return np.zeros(6, dtype=np.float32)
        try:
            object_pos = self.manipulated_object.get_pose().p
            palm_link = getattr(self, "palm_link", self.ee_link)
            palm_dist = float(np.linalg.norm(palm_link.get_pose().p - object_pos))
            if self.rm75_hand_dynamic_close_finger_mode == "min":
                fingertip_dist = float(getattr(self, "min_fingertip_obj_dist", palm_dist))
            else:
                fingertip_dist = float(getattr(self, "fingertip_obj_dist", palm_dist))
        except Exception:
            return np.zeros(6, dtype=np.float32)
        start = max(float(self.rm75_hand_dynamic_close_start), 1e-6)
        full = min(max(float(self.rm75_hand_dynamic_close_full), 0.0), start - 1e-6)
        palm_alpha = np.clip((start - palm_dist) / (start - full), 0.0, 1.0)
        finger_start = max(float(self.rm75_hand_close_min_finger_dist), full + 1e-6)
        finger_full = min(full, finger_start - 1e-6)
        finger_alpha = np.clip((finger_start - fingertip_dist) / (finger_start - finger_full), 0.0, 1.0)
        gate_mode = self.rm75_hand_dynamic_close_gate_mode.lower()
        if gate_mode == "envelope":
            alpha = float(np.sqrt(max(float(palm_alpha), 0.0) * max(float(finger_alpha), 0.0)))
        elif gate_mode == "strict":
            alpha = float(min(float(palm_alpha), float(finger_alpha)))
        else:
            palm_weight = max(float(self.rm75_hand_dynamic_close_palm_weight), 0.0)
            finger_weight = max(float(self.rm75_hand_dynamic_close_finger_weight), 0.0)
            weight_sum = max(palm_weight + finger_weight, 1e-6)
            alpha = float(np.clip((palm_weight * palm_alpha + finger_weight * finger_alpha) / weight_sum, 0.0, 1.0))
        warmup = max(int(self.rm75_hand_dynamic_close_warmup_steps), 0)
        ramp = max(int(self.rm75_hand_dynamic_close_ramp_steps), 1)
        if warmup > 0:
            phase_step = max(int(self.current_step) - int(self.pregrasp_steps) - warmup, 0)
            alpha *= float(np.clip(phase_step / ramp, 0.0, 1.0))
        if (
            not bool(getattr(self, "is_contact", False))
            and np.isfinite(self.rm75_hand_dynamic_close_drift_gate)
            and float(getattr(self, "object_xy_drift", 0.0)) > self.rm75_hand_dynamic_close_drift_gate
        ):
            alpha = min(alpha, max(float(self.rm75_hand_precontact_close_cap), 0.0) * 0.5)
        phase_weight = max(float(self.rm75_hand_dynamic_close_phase_weight), 0.0)
        if phase_weight > 0.0:
            phase_step = max(
                int(self.current_step) - int(self.pregrasp_steps) - int(self.rm75_hand_dynamic_close_phase_start),
                0,
            )
            phase_alpha = float(np.clip(phase_step / max(int(self.rm75_hand_dynamic_close_phase_ramp), 1), 0.0, 1.0))
            ref_close = float(self._reference_close_fraction())
            if min(palm_dist, fingertip_dist) <= float(self.rm75_hand_dynamic_close_phase_near_gate):
                alpha = max(alpha, phase_weight * phase_alpha * ref_close)
        if self.rm75_hand_dynamic_close_requires_contact and not bool(getattr(self, "is_contact", False)):
            alpha = 0.0
        elif not bool(getattr(self, "is_contact", False)):
            alpha = min(alpha, max(float(self.rm75_hand_precontact_close_cap), 0.0))
        if bool(getattr(self, "is_contact", False)):
            alpha = min(1.0, alpha + self.rm75_hand_dynamic_close_contact_boost)
        if int(getattr(self, "rm75_contact_hold_steps", 0)) >= max(int(self.rm75_hand_dynamic_close_contact_hold_steps), 1):
            alpha = max(alpha, float(self.rm75_hand_dynamic_close_contact_lock_alpha))
        if int(getattr(self, "rm75_lift_latch_steps_left", 0)) > 0:
            alpha = max(alpha, float(np.clip(self.rm75_post_lift_latch_hand_alpha, 0.0, 1.0)))
        smooth = float(np.clip(self.rm75_hand_dynamic_close_alpha_smooth, 0.0, 0.98))
        if smooth > 0.0:
            alpha = smooth * float(getattr(self, "rm75_last_dynamic_close_alpha", 0.0)) + (1.0 - smooth) * alpha
        self.rm75_last_dynamic_close_alpha = float(np.clip(alpha, 0.0, 1.0))
        thumb_delay = np.clip(float(self.rm75_hand_dynamic_close_thumb_delay), 0.0, 0.95)
        pinky_delay = np.clip(float(self.rm75_hand_dynamic_close_pinky_delay), 0.0, 0.95)
        group_mode = self.rm75_hand_dynamic_close_group_mode.lower()
        main_alpha = alpha
        if group_mode == "ratio":
            thumb_alpha = float(np.clip(alpha * self.rm75_hand_dynamic_close_thumb_ratio, 0.0, 1.0))
            pinky_alpha = float(np.clip(alpha * self.rm75_hand_dynamic_close_pinky_ratio, 0.0, 1.0))
        else:
            thumb_alpha = float(np.clip((alpha - thumb_delay) / max(1.0 - thumb_delay, 1e-6), 0.0, 1.0))
            pinky_alpha = float(np.clip((alpha - pinky_delay) / max(1.0 - pinky_delay, 1e-6), 0.0, 1.0))
        if bool(getattr(self, "is_contact", False)):
            thumb_alpha = max(thumb_alpha, min(1.0, alpha))
            pinky_alpha = max(pinky_alpha, min(1.0, alpha))
            if int(getattr(self, "rm75_non_thumb_contact_count", 0)) > 0:
                thumb_alpha = max(
                    thumb_alpha,
                    float(np.clip(self.rm75_hand_dynamic_close_non_thumb_thumb_alpha, 0.0, 1.0)),
                )
        else:
            thumb_alpha = min(thumb_alpha, max(float(self.rm75_hand_precontact_thumb_cap), 0.0))
            pinky_alpha = min(pinky_alpha, max(float(self.rm75_hand_precontact_pinky_cap), 0.0))
        group_alpha = np.array(
            [thumb_alpha, thumb_alpha, main_alpha, main_alpha, main_alpha, pinky_alpha],
            dtype=np.float32,
        )
        self.rm75_last_dynamic_close_group_alpha = group_alpha.copy()
        return (float(self.rm75_hand_dynamic_close_bias) * group_alpha * self.rm75_hand_close_bias_weights).astype(np.float32)

    def _hand_close_state(self) -> tuple[float, float, float, float]:
        qpos = np.asarray(self.robot.get_qpos()[self.arm_dof:], dtype=np.float32)
        qlimits = np.asarray(self.robot.get_qlimits()[self.arm_dof:], dtype=np.float32)
        if bool(getattr(self, "rm75_native_hand_control", False)):
            return self._native_close_state_from_qpos(qpos, qlimits)
        if qpos.shape[0] == 12:
            active_indices = np.array([0, 1, 5, 7, 9, 10], dtype=np.int64)
            qpos = qpos[active_indices]
            qlimits = qlimits[active_indices]
        if qpos.shape[0] != RM75_PREGRASP_HAND_QPOS.shape[0]:
            return 0.0, 0.0, 0.0, 0.0
        denom = np.maximum(qlimits[:, 1] - RM75_PREGRASP_HAND_QPOS, 1e-6)
        close = np.clip((qpos - RM75_PREGRASP_HAND_QPOS) / denom, 0.0, 1.0)
        thumb_close = float(np.mean(close[:2]))
        main_close = float(np.mean(close[2:5]))
        pinky_close = float(close[5])
        weights = np.array(
            [
                self.rm75_reward_cfg.hand_close_thumb_weight,
                self.rm75_reward_cfg.hand_close_thumb_weight,
                self.rm75_reward_cfg.hand_close_finger_weight,
                self.rm75_reward_cfg.hand_close_finger_weight,
                self.rm75_reward_cfg.hand_close_finger_weight,
                self.rm75_reward_cfg.hand_close_finger_weight,
            ],
            dtype=np.float32,
        )
        close_fraction = float(np.average(close, weights=np.maximum(weights, 1e-6)))
        return close_fraction, thumb_close, main_close, pinky_close

    def get_reward(self, action):
        if self.is_vision and not self.is_demo_rollout:
            return 0

        robot_hand_pos = self._current_fingertip_pos()
        object_pose = self.manipulated_object.get_pose()
        object_pos = object_pose.p
        object_rot = object_pose.q
        init_object_pos = self.cur_reference_motion["object_translation"][0]
        init_object_rot = self.cur_reference_motion["object_orientation"][0]

        raw_contact_groups = self._contact_groups()
        self.rm75_raw_robot_object_contact = raw_contact_groups.copy()
        self.rm75_pinky_contact = bool(raw_contact_groups.shape[0] >= 5 and raw_contact_groups[4] > 0.0)
        if bool(getattr(self, "rm75_ignore_pinky_contact", False)) and raw_contact_groups.shape[0] >= 5:
            raw_contact_groups = raw_contact_groups.copy()
            raw_contact_groups[4] = 0.0
        self.robot_object_contact[:] = raw_contact_groups
        self.is_contact = bool(np.sum(self.robot_object_contact) >= 1)
        finger_groups = self.robot_object_contact[:5] if self.robot_object_contact.shape[0] >= 5 else self.robot_object_contact
        self.rm75_thumb_contact = bool(finger_groups.shape[0] >= 1 and finger_groups[0] > 0.0)
        self.rm75_non_thumb_contact_count = int(np.sum(finger_groups[1:] > 0.0)) if finger_groups.shape[0] > 1 else 0
        self.rm75_palm_contact = bool(self.robot_object_contact.shape[0] >= 6 and self.robot_object_contact[5] > 0.0)
        raw_stable_contact = has_stable_rm75_contact(
            self.robot_object_contact,
            self.rm75_reward_cfg.required_non_thumb_contacts,
        )
        if self.is_contact:
            self.rm75_contact_hold_steps = int(getattr(self, "rm75_contact_hold_steps", 0)) + 1
        else:
            self.rm75_contact_hold_steps = 0
        if raw_stable_contact:
            self.rm75_stable_contact_hold_steps = int(getattr(self, "rm75_stable_contact_hold_steps", 0)) + 1
        else:
            self.rm75_stable_contact_hold_steps = 0
        self.object_lift = max(float(object_pos[2] - self.init_object_height), 0.0)
        self.contact_count = float(np.sum(self.robot_object_contact))
        fingertip_dists = np.linalg.norm(robot_hand_pos - object_pos[None, :], axis=1)
        self.fingertip_obj_dist = float(np.mean(fingertip_dists))
        self.min_fingertip_obj_dist = float(np.min(fingertip_dists))
        palm_link = getattr(self, "palm_link", self.ee_link)
        self.palm_obj_dist = float(np.linalg.norm(palm_link.get_pose().p - object_pos))
        (
            self.hand_close_fraction,
            self.hand_thumb_close_fraction,
            self.hand_main_close_fraction,
            self.hand_pinky_close_fraction,
        ) = self._hand_close_state()
        self.reference_close_fraction = self._reference_close_fraction()
        self.object_xy_drift_vec = np.asarray(object_pos[:2] - init_object_pos[:2], dtype=np.float32)
        self.object_xy_drift = float(np.linalg.norm(self.object_xy_drift_vec))
        object_velocity = np.zeros(3, dtype=np.float32)
        if hasattr(self.manipulated_object, "get_velocity"):
            object_velocity = np.asarray(self.manipulated_object.get_velocity(), dtype=np.float32)
        self.object_speed = float(np.linalg.norm(object_velocity))
        object_ang_velocity = np.zeros(3, dtype=np.float32)
        if hasattr(self.manipulated_object, "get_angular_velocity"):
            object_ang_velocity = np.asarray(self.manipulated_object.get_angular_velocity(), dtype=np.float32)
        self.object_ang_speed = float(np.linalg.norm(object_ang_velocity))
        self.object_tilt_err = float(rotation_distance(object_rot, init_object_rot) / np.pi)

        pregrasp_success_mode = str(getattr(self, "rm75_pregrasp_success_mode", "reference"))
        if pregrasp_success_mode == "proximity":
            self.hand_jpos_err = float(min(self.palm_obj_dist, self.min_fingertip_obj_dist))
        else:
            pregrasp_target = self._pregrasp_reference_hand_pos()
            self.hand_jpos_err = float(np.mean(np.linalg.norm(robot_hand_pos - pregrasp_target, axis=1)))

        if self.current_step <= self.pregrasp_steps:
            self.obj_com_err = 0.0
            self.obj_rot_err = 0.0
            self.hand_mjpos_err = self.hand_jpos_err
        else:
            idx = self._safe_reference_index()
            tgt_object_pos = self.cur_reference_motion["object_translation"][idx]
            tgt_object_rot = self.cur_reference_motion["object_orientation"][idx]
            tgt_robot_hand_pos = self._reference_hand_jpos(self.cur_reference_motion["robot_jpos"][idx])

            self.obj_com_err = float(np.linalg.norm(object_pos - tgt_object_pos))
            self.obj_rot_err = float(rotation_distance(object_rot, tgt_object_rot) / np.pi)
            if bool(getattr(self, "rm75_native_hand_control", False)):
                self.hand_mjpos_err = float(self.fingertip_obj_dist)
            else:
                self.hand_mjpos_err = float(np.mean(np.linalg.norm(robot_hand_pos - tgt_robot_hand_pos, axis=1)))

        result = compute_rm75_reward(
            self.rm75_reward_cfg,
            current_step=self.current_step,
            pregrasp_steps=self.pregrasp_steps,
            hand_jpos_err=self.hand_jpos_err,
            hand_mjpos_err=self.hand_mjpos_err,
            obj_com_err=self.obj_com_err,
            obj_rot_err=self.obj_rot_err,
            object_lift=self.object_lift,
            fingertip_obj_dist=self.fingertip_obj_dist,
            object_xy_drift=self.object_xy_drift,
            object_speed=self.object_speed,
            object_tilt_err=self.object_tilt_err,
            object_ang_speed=self.object_ang_speed,
            contact_groups=self.robot_object_contact,
            cartesian_error=float(self.cartesian_error),
            qvel=self.robot.get_qvel(),
            hand_close_fraction=self.hand_close_fraction,
            hand_thumb_close_fraction=self.hand_thumb_close_fraction,
            hand_main_close_fraction=self.hand_main_close_fraction,
            hand_pinky_close_fraction=self.hand_pinky_close_fraction,
            reference_close_fraction=self.reference_close_fraction,
            palm_obj_dist=self.palm_obj_dist,
            min_fingertip_obj_dist=self.min_fingertip_obj_dist,
            contact_hold_steps=self.rm75_contact_hold_steps,
            stable_contact_hold_steps=self.rm75_stable_contact_hold_steps,
        )

        pregrasp_success = result.pregrasp_success
        if pregrasp_success_mode == "proximity":
            palm_dist_thresh = float(getattr(self, "rm75_pregrasp_palm_dist_thresh", 0.16))
            min_finger_dist_thresh = float(getattr(self, "rm75_pregrasp_min_finger_dist_thresh", 0.075))
            pregrasp_success = (
                self.palm_obj_dist <= palm_dist_thresh
                or self.min_fingertip_obj_dist <= min_finger_dist_thresh
            )
        if not self.pregrasp_success and self.current_step == self.pregrasp_steps and pregrasp_success:
            self.pregrasp_success = True
        self.stable_grasp_contact = result.stable_grasp_contact
        self.contact_count = result.contact_count
        return result.reward

    def get_info(self):
        info = super().get_info()
        info["stable_grasp_contact"] = bool(self.stable_grasp_contact)
        info["contact_count"] = float(self.contact_count)
        info["fingertip_obj_dist"] = float(getattr(self, "fingertip_obj_dist", 0.0))
        info["min_fingertip_obj_dist"] = float(getattr(self, "min_fingertip_obj_dist", 0.0))
        info["palm_obj_dist"] = float(getattr(self, "palm_obj_dist", 0.0))
        info["hand_close_fraction"] = float(getattr(self, "hand_close_fraction", 0.0))
        info["hand_thumb_close_fraction"] = float(getattr(self, "hand_thumb_close_fraction", 0.0))
        info["hand_main_close_fraction"] = float(getattr(self, "hand_main_close_fraction", 0.0))
        info["hand_pinky_close_fraction"] = float(getattr(self, "hand_pinky_close_fraction", 0.0))
        info["reference_close_fraction"] = float(getattr(self, "reference_close_fraction", 0.0))
        info["reference_thumb_close_fraction"] = float(getattr(self, "reference_thumb_close_fraction", 0.0))
        info["reference_main_close_fraction"] = float(getattr(self, "reference_main_close_fraction", 0.0))
        info["reference_pinky_close_fraction"] = float(getattr(self, "reference_pinky_close_fraction", 0.0))
        info["hand_reference_close_error"] = abs(
            float(getattr(self, "hand_close_fraction", 0.0))
            - float(getattr(self, "reference_close_fraction", 0.0))
        )
        info["pregrasp_safe_active"] = bool(getattr(self, "rm75_pregrasp_safe_active", False))
        info["effective_arm_action_scale"] = float(getattr(self, "rm75_last_effective_arm_scale", 0.0))
        info["effective_hand_action_scale"] = float(getattr(self, "rm75_last_effective_hand_scale", 0.0))
        info["project_to_object_active"] = bool(getattr(self, "rm75_last_project_to_object_active", False))
        info["post_stable_lift_only_active"] = bool(
            getattr(self, "rm75_last_post_stable_lift_only_active", False)
        )
        info["post_contact_lift_assist_active"] = bool(
            getattr(self, "rm75_last_post_contact_lift_assist_active", False)
        )
        info["post_contact_wrist_bias_active"] = bool(
            getattr(self, "rm75_last_post_contact_wrist_bias_active", False)
        )
        info["scripted_lift_prior_active"] = bool(
            getattr(self, "rm75_last_scripted_lift_prior_active", False)
        )
        info["lift_latch_steps_left"] = int(getattr(self, "rm75_lift_latch_steps_left", 0))
        info["done_on_norm_success_10_active"] = bool(
            getattr(self, "rm75_done_on_norm_success_10_active", False)
        )
        info["dynamic_close_alpha"] = float(getattr(self, "rm75_last_dynamic_close_alpha", 0.0))
        group_alpha = np.asarray(getattr(self, "rm75_last_dynamic_close_group_alpha", np.zeros(6)), dtype=np.float32)
        info["dynamic_close_thumb_alpha"] = float(np.mean(group_alpha[:2])) if group_alpha.shape[0] >= 2 else 0.0
        info["dynamic_close_main_alpha"] = float(np.mean(group_alpha[2:5])) if group_alpha.shape[0] >= 5 else 0.0
        info["dynamic_close_pinky_alpha"] = float(group_alpha[5]) if group_alpha.shape[0] >= 6 else 0.0
        info["contact_hold_steps"] = int(getattr(self, "rm75_contact_hold_steps", 0))
        info["stable_contact_hold_steps"] = int(getattr(self, "rm75_stable_contact_hold_steps", 0))
        info["thumb_contact"] = bool(getattr(self, "rm75_thumb_contact", False))
        info["non_thumb_contact_count"] = int(getattr(self, "rm75_non_thumb_contact_count", 0))
        info["palm_contact"] = bool(getattr(self, "rm75_palm_contact", False))
        info["pinky_contact"] = bool(getattr(self, "rm75_pinky_contact", False))
        info["ignore_pinky_contact"] = bool(getattr(self, "rm75_ignore_pinky_contact", False))
        info["object_xy_drift"] = float(getattr(self, "object_xy_drift", 0.0))
        drift_vec = np.asarray(getattr(self, "object_xy_drift_vec", np.zeros(2)), dtype=np.float32)
        info["object_xy_drift_x"] = float(drift_vec[0]) if drift_vec.shape[0] >= 1 else 0.0
        info["object_xy_drift_y"] = float(drift_vec[1]) if drift_vec.shape[0] >= 2 else 0.0
        info["object_speed"] = float(getattr(self, "object_speed", 0.0))
        info["object_tilt_err"] = float(getattr(self, "object_tilt_err", 0.0))
        info["object_ang_speed"] = float(getattr(self, "object_ang_speed", 0.0))
        contact_success = bool(
            getattr(self, "rm75_thumb_contact", False)
            and int(getattr(self, "rm75_non_thumb_contact_count", 0))
            >= int(getattr(self, "rm75_success_min_non_thumb_contacts", 1))
        )
        lift_success_5cm = float(getattr(self, "object_lift", 0.0)) > 0.05
        lift_success_target = float(getattr(self, "object_lift", 0.0)) > float(
            getattr(self, "rm75_norm_success_lift_thresh", 0.05)
        )
        object_pos = np.asarray(self.manipulated_object.get_pose().p, dtype=np.float32)
        if self.norm_traj:
            norm_goal = np.array([self.init_x, self.init_y, 0.2], dtype=np.float32)
            norm_success_3 = np.linalg.norm(object_pos - norm_goal) < 0.03
            norm_success_10 = lift_success_target
        elif self.is_vision:
            target = np.asarray(self.target_object_pos, dtype=np.float32)
            norm_success_3 = np.linalg.norm(object_pos - target) < 0.03
            norm_success_10 = np.linalg.norm(object_pos - target) < 0.1
        else:
            target = np.asarray(self.target_object.get_pose().p, dtype=np.float32)
            norm_success_3 = np.linalg.norm(object_pos - target) < 0.03
            norm_success_10 = np.linalg.norm(object_pos - target) < 0.1
        info["contact_success"] = bool(contact_success)
        info["lift_success_5cm"] = bool(lift_success_5cm)
        info["lift_success_target"] = bool(lift_success_target)
        contact_gate = bool(contact_success) or not bool(getattr(self, "rm75_norm_success_requires_contact", True))
        info["norm_success_3"] = bool(contact_gate and norm_success_3)
        info["norm_success_10"] = bool(contact_gate and norm_success_10)
        grasp_score, precision_score = compute_rm75_grasp_scores(
            object_lift=getattr(self, "object_lift", 0.0),
            contact_count=getattr(self, "contact_count", 0.0),
            stable_grasp_contact=getattr(self, "stable_grasp_contact", False),
            contact_hold_steps=getattr(self, "rm75_contact_hold_steps", 0),
            stable_contact_hold_steps=getattr(self, "rm75_stable_contact_hold_steps", 0),
            thumb_contact=getattr(self, "rm75_thumb_contact", False),
            non_thumb_contact_count=getattr(self, "rm75_non_thumb_contact_count", 0),
            object_xy_drift=getattr(self, "object_xy_drift", 0.0),
            object_speed=getattr(self, "object_speed", 0.0),
            object_tilt_err=getattr(self, "object_tilt_err", 0.0),
            object_ang_speed=getattr(self, "object_ang_speed", 0.0),
            lift_target=getattr(self, "rm75_score_lift_target", 0.08),
        )
        info["rm75_grasp_score"] = grasp_score
        info["rm75_precision_score"] = precision_score
        info["rm75_task_score"] = compute_rm75_task_score(
            obj_com_err=getattr(self, "obj_com_err", 0.0),
            obj_rot_err=getattr(self, "obj_rot_err", 0.0),
            object_lift=getattr(self, "object_lift", 0.0),
            stable_grasp_contact=getattr(self, "stable_grasp_contact", False),
            stable_contact_hold_steps=getattr(self, "rm75_stable_contact_hold_steps", 0),
            thumb_contact=getattr(self, "rm75_thumb_contact", False),
            non_thumb_contact_count=getattr(self, "rm75_non_thumb_contact_count", 0),
            object_xy_drift=getattr(self, "object_xy_drift", 0.0),
            object_speed=getattr(self, "object_speed", 0.0),
            object_tilt_err=getattr(self, "object_tilt_err", 0.0),
            object_ang_speed=getattr(self, "object_ang_speed", 0.0),
            lift_target=getattr(self, "rm75_score_lift_target", 0.08),
        )
        return info

    def is_done(self):
        if self.is_vision and not self.is_demo_rollout:
            return self.current_step >= self.imitate_steps

        if not self.pregrasp_success:
            if self.current_step >= self.pregrasp_steps:
                if bool(getattr(self, "rm75_done_on_pregrasp_failure", True)):
                    return True
                self.pregrasp_success = True
            else:
                self.traj_step += 1
                return False

        contact_done, self.no_contact_steps = should_terminate_for_contact_loss(
            self.robot_name,
            bool(self.is_contact),
            self.current_step,
            self.pregrasp_steps,
            getattr(self, "no_contact_steps", 0),
            grace_steps=self.rm75_reward_cfg.no_contact_grace_steps,
        )
        if bool(getattr(self, "rm75_done_on_norm_success_10", False)):
            contact_success = bool(
                getattr(self, "rm75_thumb_contact", False)
                and int(getattr(self, "rm75_non_thumb_contact_count", 0))
                >= int(getattr(self, "rm75_success_min_non_thumb_contacts", 1))
            )
            lift_success = float(getattr(self, "object_lift", 0.0)) > float(
                getattr(self, "rm75_norm_success_lift_thresh", 0.05)
            )
            contact_gate = contact_success or not bool(getattr(self, "rm75_norm_success_requires_contact", True))
            if contact_gate and lift_success:
                self.rm75_done_on_norm_success_10_active = True
                return True
        if self.obj_com_err >= self.rm75_reward_cfg.obj_com_done_thresh or contact_done:
            return True
        if self.rm75_reward_cfg.bad_push_done and not bool(getattr(self, "stable_grasp_contact", False)):
            post_pregrasp_step = int(self.current_step) - int(self.pregrasp_steps)
            bad_drift = float(getattr(self, "object_xy_drift", 0.0)) >= self.rm75_reward_cfg.bad_push_drift_thresh
            bad_tilt = float(getattr(self, "object_tilt_err", 0.0)) >= self.rm75_reward_cfg.bad_push_tilt_thresh
            if post_pregrasp_step >= int(self.rm75_reward_cfg.bad_push_min_step) and (bad_drift or bad_tilt):
                return True
        if self.current_step >= self.imitate_steps:
            return True

        self.traj_step += 1
        return False
