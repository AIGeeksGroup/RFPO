from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class RM75RewardConfig:
    pregrasp_pos_scale: float = 10.0
    pregrasp_reward_scale: float = 10.0
    pregrasp_success_thresh: float = 0.075
    finger_approach_reward_scale: float = 1.5
    finger_approach_scale: float = 35.0
    min_finger_approach_reward_scale: float = 0.0
    min_finger_approach_scale: float = 25.0
    palm_approach_reward_scale: float = 0.0
    palm_approach_scale: float = 12.0
    contact_reward_scale: float = 0.5
    thumb_contact_reward_scale: float = 0.75
    non_thumb_contact_reward_scale: float = 0.35
    stable_contact_bonus: float = 0.75
    contact_hold_bonus_scale: float = 0.0
    contact_hold_bonus_steps: int = 6
    required_non_thumb_contacts: int = 1
    no_contact_penalty: float = 0.25
    object_reward_requires_contact: bool = True
    object_reward_requires_stable_contact: bool = False
    object_reward_contact_hold_steps: int = 0
    object_reward_scale: float = 10.0
    obj_err_scale: float = 50.0
    obj_rot_term: float = 0.1
    hand_mimic_reward_scale: float = 4.0
    hand_mimic_scale: float = 10.0
    hand_close_reward_scale: float = 0.0
    hand_close_near_dist: float = 0.08
    hand_close_palm_near_dist: float = 0.22
    hand_close_min_finger_near_dist: float = 0.12
    hand_close_contact_only: bool = False
    hand_close_thumb_weight: float = 1.0
    hand_close_finger_weight: float = 1.0
    hand_close_target: float = 0.75
    hand_open_penalty_scale: float = 0.0
    thumb_close_reward_scale: float = 0.0
    thumb_open_penalty_scale: float = 0.0
    thumb_close_target: float = 0.55
    reference_close_reward_scale: float = 0.0
    reference_close_penalty_scale: float = 0.0
    reference_close_tolerance: float = 0.08
    reference_close_start_step: int = 0
    early_close_penalty_scale: float = 0.0
    early_close_palm_dist: float = 0.20
    early_close_min_finger_dist: float = 0.10
    hand_synergy_reward_scale: float = 0.0
    hand_synergy_penalty_scale: float = 0.0
    hand_synergy_thumb_target_ratio: float = 0.75
    hand_synergy_pinky_target_ratio: float = 0.80
    hand_synergy_min_main_close: float = 0.0
    hand_synergy_balance_penalty_scale: float = 0.0
    lift_bonus_thresh: float = 0.02
    lift_bonus_mag: float = 2.5
    lift_reward_scale: float = 20.0
    lift_reward_cap: float = 0.08
    object_xy_drift_free_thresh: float = 0.015
    object_xy_drift_penalty_scale: float = 15.0
    object_speed_penalty_scale: float = 0.0
    object_tilt_free_thresh: float = 0.20
    object_tilt_penalty_scale: float = 0.0
    object_ang_vel_penalty_scale: float = 0.0
    unstable_push_penalty_scale: float = 0.0
    controller_penalty_scale: float = 1e3
    action_penalty_scale: float = 0.01
    reward_divisor: float = 10.0
    obj_com_done_thresh: float = 0.18
    no_contact_grace_steps: int = 30
    bad_push_done: bool = False
    bad_push_min_step: int = 6
    bad_push_drift_thresh: float = 0.055
    bad_push_tilt_thresh: float = 0.42

    @classmethod
    def from_kwargs(cls, kwargs: dict | None) -> "RM75RewardConfig":
        if kwargs is None:
            return cls()
        known = {field.name for field in fields(cls)}
        values = {key: value for key, value in dict(kwargs).items() if key in known}
        return cls(**values)


@dataclass(frozen=True)
class RM75RewardResult:
    reward: float
    pregrasp_success: bool
    stable_grasp_contact: bool
    contact_count: float


def has_stable_rm75_contact(contact_groups: np.ndarray, required_non_thumb_contacts: int = 1) -> bool:
    contact_groups = np.asarray(contact_groups, dtype=np.float32)
    if contact_groups.shape[0] < 2:
        return False
    finger_groups = contact_groups[:5]
    thumb_contact = bool(finger_groups[0] > 0.0)
    non_thumb_contacts = int(np.sum(finger_groups[1:] > 0.0))
    return thumb_contact and non_thumb_contacts >= int(required_non_thumb_contacts)


def compute_rm75_reward(
    cfg: RM75RewardConfig,
    *,
    current_step: int,
    pregrasp_steps: int,
    hand_jpos_err: float,
    hand_mjpos_err: float,
    obj_com_err: float,
    obj_rot_err: float,
    object_lift: float,
    fingertip_obj_dist: float = 0.0,
    object_xy_drift: float = 0.0,
    object_speed: float = 0.0,
    object_tilt_err: float = 0.0,
    object_ang_speed: float = 0.0,
    contact_groups: np.ndarray,
    cartesian_error: float,
    qvel: np.ndarray,
    hand_close_fraction: float = 0.0,
    hand_thumb_close_fraction: float = 0.0,
    hand_main_close_fraction: float = 0.0,
    hand_pinky_close_fraction: float = 0.0,
    reference_close_fraction: float = 0.0,
    palm_obj_dist: float | None = None,
    min_fingertip_obj_dist: float | None = None,
    contact_hold_steps: int = 0,
    stable_contact_hold_steps: int = 0,
) -> RM75RewardResult:
    contact_groups = np.asarray(contact_groups, dtype=np.float32)
    qvel = np.asarray(qvel, dtype=np.float32)
    contact_count = float(np.sum(contact_groups))
    stable_contact = has_stable_rm75_contact(contact_groups, cfg.required_non_thumb_contacts)

    if current_step <= pregrasp_steps:
        reward = cfg.pregrasp_reward_scale * np.exp(-cfg.pregrasp_pos_scale * float(hand_jpos_err))
    else:
        has_contact = contact_count > 0.0
        if cfg.object_reward_requires_stable_contact:
            object_reward_gate = stable_contact
        else:
            object_reward_gate = (has_contact or not cfg.object_reward_requires_contact)
        hold_gate = 1.0
        if cfg.object_reward_contact_hold_steps > 0:
            reward_hold_steps = (
                stable_contact_hold_steps
                if cfg.object_reward_requires_stable_contact
                else contact_hold_steps
            )
            hold_gate = min(
                max(int(reward_hold_steps), 0) / max(int(cfg.object_reward_contact_hold_steps), 1),
                1.0,
            )
            if not object_reward_gate:
                hold_gate = 0.0
        finger_groups = contact_groups[:5] if contact_groups.shape[0] >= 5 else contact_groups
        thumb_contact = float(finger_groups[0] > 0.0) if finger_groups.shape[0] >= 1 else 0.0
        non_thumb_contacts = float(np.sum(finger_groups[1:] > 0.0)) if finger_groups.shape[0] > 1 else 0.0

        palm_obj_dist = float("inf") if palm_obj_dist is None else float(palm_obj_dist)
        min_fingertip_obj_dist = (
            float("inf") if min_fingertip_obj_dist is None else float(min_fingertip_obj_dist)
        )

        reward = cfg.finger_approach_reward_scale * np.exp(
            -cfg.finger_approach_scale * float(fingertip_obj_dist)
        )
        reward += cfg.min_finger_approach_reward_scale * np.exp(
            -cfg.min_finger_approach_scale * min_fingertip_obj_dist
        )
        reward += cfg.palm_approach_reward_scale * np.exp(-cfg.palm_approach_scale * palm_obj_dist)
        reward += cfg.contact_reward_scale * contact_count
        reward += cfg.thumb_contact_reward_scale * thumb_contact
        reward += cfg.non_thumb_contact_reward_scale * non_thumb_contacts
        if stable_contact:
            reward += cfg.stable_contact_bonus
        if contact_count > 0.0 and cfg.contact_hold_bonus_scale > 0.0:
            hold_frac = min(max(int(contact_hold_steps), 0), max(int(cfg.contact_hold_bonus_steps), 1))
            reward += cfg.contact_hold_bonus_scale * (hold_frac / max(int(cfg.contact_hold_bonus_steps), 1))
        obj_err = float(obj_com_err) + cfg.obj_rot_term * float(obj_rot_err)
        if object_reward_gate:
            reward += hold_gate * cfg.object_reward_scale * np.exp(-cfg.obj_err_scale * obj_err)
        else:
            reward -= cfg.no_contact_penalty
        reward += cfg.hand_mimic_reward_scale * np.exp(-cfg.hand_mimic_scale * float(hand_mjpos_err))
        close_near = (
            float(fingertip_obj_dist) <= cfg.hand_close_near_dist
            or palm_obj_dist <= cfg.hand_close_palm_near_dist
            or min_fingertip_obj_dist <= cfg.hand_close_min_finger_near_dist
        )
        close_gate = close_near and (has_contact or not cfg.hand_close_contact_only)
        if close_gate:
            close_fraction = float(np.clip(hand_close_fraction, 0.0, 1.0))
            reward += cfg.hand_close_reward_scale * close_fraction
            open_gap = max(float(cfg.hand_close_target) - close_fraction, 0.0)
            reward -= cfg.hand_open_penalty_scale * open_gap
            post_pregrasp_step = int(current_step) - int(pregrasp_steps)
            if post_pregrasp_step >= int(cfg.reference_close_start_step):
                ref_close = float(np.clip(reference_close_fraction, 0.0, 1.0))
                ref_error = abs(close_fraction - ref_close)
                reward += cfg.reference_close_reward_scale * np.exp(-6.0 * ref_error)
                under_close_gap = max(ref_close - close_fraction - float(cfg.reference_close_tolerance), 0.0)
                reward -= cfg.reference_close_penalty_scale * under_close_gap
            main_close = float(np.clip(hand_main_close_fraction, 0.0, 1.0))
            thumb_close = float(np.clip(hand_thumb_close_fraction, 0.0, 1.0))
            pinky_close = float(np.clip(hand_pinky_close_fraction, 0.0, 1.0))
            reward += cfg.thumb_close_reward_scale * thumb_close
            thumb_open_gap = max(float(cfg.thumb_close_target) - thumb_close, 0.0)
            reward -= cfg.thumb_open_penalty_scale * thumb_open_gap
            thumb_target = cfg.hand_synergy_thumb_target_ratio * main_close
            pinky_target = cfg.hand_synergy_pinky_target_ratio * main_close
            synergy_error = abs(thumb_close - thumb_target) + abs(pinky_close - pinky_target)
            if main_close >= cfg.hand_synergy_min_main_close:
                reward += cfg.hand_synergy_reward_scale * np.exp(-4.0 * synergy_error)
            reward -= cfg.hand_synergy_penalty_scale * synergy_error
            group_spread = max(thumb_close, main_close, pinky_close) - min(thumb_close, main_close, pinky_close)
            reward -= cfg.hand_synergy_balance_penalty_scale * group_spread
        else:
            close_fraction = float(np.clip(hand_close_fraction, 0.0, 1.0))
            too_far_to_close = (
                palm_obj_dist > cfg.early_close_palm_dist
                and min_fingertip_obj_dist > cfg.early_close_min_finger_dist
            )
            if too_far_to_close:
                reward -= cfg.early_close_penalty_scale * close_fraction
        lift_gate = stable_contact if cfg.object_reward_requires_stable_contact else object_reward_gate
        if lift_gate and float(object_lift) > cfg.lift_bonus_thresh:
            reward += hold_gate * cfg.lift_bonus_mag
        if lift_gate:
            reward += hold_gate * cfg.lift_reward_scale * min(float(object_lift), cfg.lift_reward_cap)

        xy_excess = max(float(object_xy_drift) - cfg.object_xy_drift_free_thresh, 0.0)
        reward -= cfg.object_xy_drift_penalty_scale * xy_excess
        reward -= cfg.object_speed_penalty_scale * float(object_speed)
        tilt_excess = max(float(object_tilt_err) - cfg.object_tilt_free_thresh, 0.0)
        reward -= cfg.object_tilt_penalty_scale * tilt_excess
        reward -= cfg.object_ang_vel_penalty_scale * float(object_ang_speed)
        if not stable_contact:
            unstable_push = xy_excess + 0.5 * tilt_excess
            reward -= cfg.unstable_push_penalty_scale * unstable_push

    controller_penalty = -(float(cartesian_error) ** 2) * cfg.controller_penalty_scale
    action_penalty = -cfg.action_penalty_scale * float(np.sum(np.clip(qvel, -1.0, 1.0) ** 2))
    reward = (reward + controller_penalty + action_penalty) / cfg.reward_divisor

    return RM75RewardResult(
        reward=float(reward),
        pregrasp_success=float(hand_jpos_err) < cfg.pregrasp_success_thresh,
        stable_grasp_contact=stable_contact,
        contact_count=contact_count,
    )
