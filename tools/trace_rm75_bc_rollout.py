#!/usr/bin/env python3
"""Trace RM75 BC policy rollouts against the scripted expert action."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

import _init_paths  # noqa: F401

from algos.rl.fpo_core import FPOPolicyConfig, FPOStatePolicy
from hand_imitation.env.gym_wrapper import GymWrapper
from tools.scripted_rm75_grasp_smoke import (
    _approach_lift_action,
    _approach_target_action,
    apply_approach_pregrasp_reference,
)


DEFAULT_SEQ = "ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--rollout", choices=("policy", "scripted"), default="policy")
    parser.add_argument("--scripted-mode", choices=("approach_lift", "approach_target"), default="approach_target")
    parser.add_argument("--seq-name", default=DEFAULT_SEQ)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--reset-settle-steps", type=int, default=0)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--robot-base-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--robot-base-rpy", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--robot-base-yaw", type=float, default=0.0)
    parser.add_argument("--bad-push-done", action="store_true")
    parser.add_argument("--is-eval", action="store_true")
    parser.add_argument("--skip-explicit-approach-apply", action="store_true")
    parser.add_argument("--no-done-on-pregrasp-failure", action="store_true")
    parser.add_argument("--approach-delta", nargs=3, type=float, default=[0.01, -0.06, -0.02])
    parser.add_argument("--approach-steps", type=int, default=14)
    parser.add_argument("--close-steps", type=int, default=30)
    parser.add_argument("--hold-steps", type=int, default=36)
    parser.add_argument("--lift-steps", type=int, default=140)
    parser.add_argument("--lift-height", type=float, default=0.14)
    parser.add_argument("--target-track-max-xy-step", type=float, default=0.006)
    parser.add_argument("--target-track-max-z-step", type=float, default=0.005)
    parser.add_argument("--target-track-gain", type=float, default=0.65)
    parser.add_argument("--target-track-z-deadband", type=float, default=0.004)
    parser.add_argument("--target-track-start-step", type=int, default=None)
    parser.add_argument("--target-track-require-stable", action="store_true")
    parser.add_argument("--close-val", type=float, default=0.30)
    parser.add_argument("--thumb-yaw", type=float, default=0.90)
    parser.add_argument("--thumb-pitch", type=float, default=0.85)
    parser.add_argument("--finger-close", type=float, default=0.75)
    parser.add_argument("--pinky-close", type=float, default=None)
    parser.add_argument("--angular-action", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--angular-start-step", type=int, default=0)
    parser.add_argument("--angular-end-step", type=int, default=0)
    parser.add_argument("--ignore-pinky-contact", action="store_true")
    parser.add_argument("--disable-pinky-action", action="store_true")
    parser.add_argument("--pinky-action-value", type=float, default=0.0)
    parser.add_argument("--hand-close-bias", type=float, default=0.0)
    parser.add_argument("--hand-phase-close-bias", type=float, default=0.0)
    parser.add_argument("--hand-phase-close-start-step", type=int, default=5)
    parser.add_argument("--hand-phase-close-ramp-steps", type=int, default=18)
    parser.add_argument("--hand-close-bias-weights", nargs=6, type=float, default=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    parser.add_argument("--thumb-action-cap-until-non-thumb", type=float, default=None)
    parser.add_argument("--thumb-action-cap-release-non-thumb-contacts", type=int, default=2)
    parser.add_argument("--thumb-action-cap-release-contact-hold-steps", type=int, default=1)
    parser.add_argument("--success-min-non-thumb-contacts", type=int, default=1)
    parser.add_argument("--required-non-thumb-contacts", type=int, default=1)
    parser.add_argument("--no-contact-grace-steps", type=int, default=60)
    parser.add_argument("--force-imitate-steps", type=int, default=205)
    parser.add_argument("--norm-success-lift-thresh", type=float, default=0.05)
    parser.add_argument("--score-lift-target", type=float, default=0.08)
    parser.add_argument("--no-done-on-norm-success", action="store_true")
    parser.add_argument("--post-pregrasp-hold-until-stable", action="store_true")
    parser.add_argument("--post-pregrasp-hold-arm-scale", type=float, default=0.012)
    parser.add_argument("--post-pregrasp-required-stable-hold-steps", type=int, default=2)
    parser.add_argument("--post-pregrasp-project-to-object", action="store_true")
    parser.add_argument("--post-pregrasp-project-tangent-scale", type=float, default=0.02)
    parser.add_argument("--post-pregrasp-project-max-approach-speed", type=float, default=0.018)
    parser.add_argument("--post-pregrasp-project-max-retreat-speed", type=float, default=0.004)
    parser.add_argument("--post-pregrasp-project-approach-bias", type=float, default=0.006)
    parser.add_argument("--scripted-lift-prior", action="store_true")
    parser.add_argument("--scripted-lift-prior-start-step", type=int, default=0)
    parser.add_argument("--scripted-lift-prior-approach-steps", type=int, default=0)
    parser.add_argument("--scripted-lift-prior-close-steps", type=int, default=18)
    parser.add_argument("--scripted-lift-prior-hold-steps", type=int, default=10)
    parser.add_argument("--scripted-lift-prior-lift-steps", type=int, default=45)
    parser.add_argument("--scripted-lift-prior-lift-height", type=float, default=0.10)
    parser.add_argument("--scripted-lift-prior-pos-gain", type=float, default=0.04)
    parser.add_argument("--scripted-lift-prior-blend", type=float, default=1.0)
    parser.add_argument("--scripted-lift-prior-track-mode", choices=("initial", "current"), default="initial")
    parser.add_argument("--scripted-lift-prior-allow-angular", action="store_true")
    parser.add_argument("--scripted-hand-prior", action="store_true")
    parser.add_argument("--scripted-hand-prior-blend", type=float, default=1.0)
    parser.add_argument("--scripted-hand-prior-approach-steps", type=int, default=14)
    parser.add_argument("--scripted-hand-prior-close-steps", type=int, default=30)
    parser.add_argument("--scripted-hand-prior-thumb-yaw", type=float, default=0.90)
    parser.add_argument("--scripted-hand-prior-thumb-pitch", type=float, default=0.85)
    parser.add_argument("--scripted-hand-prior-finger-close", type=float, default=0.75)
    parser.add_argument("--scripted-hand-prior-pinky-close", type=float, default=0.75)
    parser.add_argument("--scripted-action-prior", action="store_true")
    parser.add_argument("--scripted-action-prior-blend", type=float, default=1.0)
    parser.add_argument("--scripted-action-prior-approach-steps", type=int, default=14)
    parser.add_argument("--scripted-action-prior-close-steps", type=int, default=30)
    parser.add_argument("--scripted-action-prior-hold-steps", type=int, default=36)
    parser.add_argument("--scripted-action-prior-lift-steps", type=int, default=140)
    parser.add_argument("--scripted-action-prior-lift-height", type=float, default=0.18)
    parser.add_argument("--scripted-action-prior-pos-gain", type=float, default=0.04)
    parser.add_argument("--scripted-action-prior-thumb-yaw", type=float, default=0.90)
    parser.add_argument("--scripted-action-prior-thumb-pitch", type=float, default=0.85)
    parser.add_argument("--scripted-action-prior-finger-close", type=float, default=0.75)
    parser.add_argument("--scripted-action-prior-pinky-close", type=float, default=0.75)
    parser.add_argument("--post-stable-lift-only", action="store_true")
    parser.add_argument("--post-stable-max-xy-speed", type=float, default=0.012)
    parser.add_argument("--post-stable-max-down-speed", type=float, default=0.002)
    parser.add_argument("--post-stable-max-up-speed", type=float, default=0.035)
    parser.add_argument("--post-stable-lift-bias", type=float, default=0.006)
    parser.add_argument("--post-stable-xy-drift-correction-gain", type=float, default=0.0)
    parser.add_argument("--post-stable-xy-drift-correction-max-speed", type=float, default=0.0)
    parser.add_argument("--post-contact-lift-assist", action="store_true")
    parser.add_argument("--post-contact-min-hold-steps", type=int, default=2)
    parser.add_argument("--post-contact-min-non-thumb-contacts", type=int, default=1)
    parser.add_argument("--post-contact-max-xy-speed", type=float, default=0.004)
    parser.add_argument("--post-contact-max-down-speed", type=float, default=0.002)
    parser.add_argument("--post-contact-max-up-speed", type=float, default=0.045)
    parser.add_argument("--post-contact-lift-bias", type=float, default=0.012)
    parser.add_argument("--post-contact-xy-drift-correction-gain", type=float, default=0.0)
    parser.add_argument("--post-contact-xy-drift-correction-max-speed", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_policy(path: Path, device: torch.device) -> FPOStatePolicy:
    checkpoint = torch.load(path, map_location=device)
    cfg = FPOPolicyConfig.from_mapping(checkpoint["policy_config"])
    policy = FPOStatePolicy(cfg).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def _task_kwargs(args: argparse.Namespace) -> dict:
    return {
        "action": "relocate",
        "rm75_native_hand_control": True,
        "rm75_enforce_native_mimic_qpos": False,
        "rm75_pregrasp_success_mode": "proximity",
        "rm75_robot_base_offset": [float(v) for v in args.robot_base_offset],
        "rm75_robot_base_rpy": [float(v) for v in args.robot_base_rpy],
        "rm75_robot_base_yaw": float(args.robot_base_yaw),
        "rm75_pregrasp_palm_dist_thresh": 0.22,
        "rm75_pregrasp_min_finger_dist_thresh": 0.10,
        "rm75_done_on_pregrasp_failure": not bool(args.no_done_on_pregrasp_failure),
        "rm75_native_apply_template_approach_pregrasp": True,
        "object_scale": float(args.object_scale),
        "rm75_default_approach_delta": [float(v) for v in args.approach_delta],
        "rm75_override_template_approach_delta": True,
        "rm75_reset_settle_steps": int(args.reset_settle_steps),
        "rm75_force_imitate_steps": int(args.force_imitate_steps),
        "rm75_done_on_norm_success_10": not bool(args.no_done_on_norm_success),
        "rm75_norm_success_lift_thresh": float(args.norm_success_lift_thresh),
        "rm75_norm_success_requires_contact": True,
        "rm75_success_min_non_thumb_contacts": int(args.success_min_non_thumb_contacts),
        "rm75_score_lift_target": float(args.score_lift_target),
        "rm75_ignore_pinky_contact": bool(args.ignore_pinky_contact),
        "rm75_disable_pinky_action": bool(args.disable_pinky_action),
        "rm75_pinky_action_value": float(args.pinky_action_value),
        "rm75_hand_close_bias": float(args.hand_close_bias),
        "rm75_hand_phase_close_bias": float(args.hand_phase_close_bias),
        "rm75_hand_phase_close_start_step": int(args.hand_phase_close_start_step),
        "rm75_hand_phase_close_ramp_steps": int(args.hand_phase_close_ramp_steps),
        "rm75_hand_close_bias_weights": [float(v) for v in args.hand_close_bias_weights],
        "rm75_thumb_action_cap_until_non_thumb": (
            None if args.thumb_action_cap_until_non_thumb is None else float(args.thumb_action_cap_until_non_thumb)
        ),
        "rm75_thumb_action_cap_release_non_thumb_contacts": int(args.thumb_action_cap_release_non_thumb_contacts),
        "rm75_thumb_action_cap_release_contact_hold_steps": int(args.thumb_action_cap_release_contact_hold_steps),
        "rm75_post_pregrasp_hold_until_stable": bool(args.post_pregrasp_hold_until_stable),
        "rm75_post_pregrasp_hold_arm_scale": float(args.post_pregrasp_hold_arm_scale),
        "rm75_post_pregrasp_required_stable_hold_steps": int(args.post_pregrasp_required_stable_hold_steps),
        "rm75_post_pregrasp_project_to_object": bool(args.post_pregrasp_project_to_object),
        "rm75_post_pregrasp_project_until_stable": True,
        "rm75_post_pregrasp_project_tangent_scale": float(args.post_pregrasp_project_tangent_scale),
        "rm75_post_pregrasp_project_max_approach_speed": float(args.post_pregrasp_project_max_approach_speed),
        "rm75_post_pregrasp_project_max_retreat_speed": float(args.post_pregrasp_project_max_retreat_speed),
        "rm75_post_pregrasp_project_approach_bias": float(args.post_pregrasp_project_approach_bias),
        "rm75_scripted_lift_prior": bool(args.scripted_lift_prior),
        "rm75_scripted_lift_prior_start_step": int(args.scripted_lift_prior_start_step),
        "rm75_scripted_lift_prior_approach_steps": int(args.scripted_lift_prior_approach_steps),
        "rm75_scripted_lift_prior_close_steps": int(args.scripted_lift_prior_close_steps),
        "rm75_scripted_lift_prior_hold_steps": int(args.scripted_lift_prior_hold_steps),
        "rm75_scripted_lift_prior_lift_steps": int(args.scripted_lift_prior_lift_steps),
        "rm75_scripted_lift_prior_lift_height": float(args.scripted_lift_prior_lift_height),
        "rm75_scripted_lift_prior_pos_gain": float(args.scripted_lift_prior_pos_gain),
        "rm75_scripted_lift_prior_blend": float(args.scripted_lift_prior_blend),
        "rm75_scripted_lift_prior_track_mode": str(args.scripted_lift_prior_track_mode),
        "rm75_scripted_lift_prior_zero_angular": not bool(args.scripted_lift_prior_allow_angular),
        "rm75_scripted_hand_prior": bool(args.scripted_hand_prior),
        "rm75_scripted_hand_prior_blend": float(args.scripted_hand_prior_blend),
        "rm75_scripted_hand_prior_approach_steps": int(args.scripted_hand_prior_approach_steps),
        "rm75_scripted_hand_prior_close_steps": int(args.scripted_hand_prior_close_steps),
        "rm75_scripted_hand_prior_thumb_yaw": float(args.scripted_hand_prior_thumb_yaw),
        "rm75_scripted_hand_prior_thumb_pitch": float(args.scripted_hand_prior_thumb_pitch),
        "rm75_scripted_hand_prior_finger_close": float(args.scripted_hand_prior_finger_close),
        "rm75_scripted_hand_prior_pinky_close": float(args.scripted_hand_prior_pinky_close),
        "rm75_scripted_action_prior": bool(args.scripted_action_prior),
        "rm75_scripted_action_prior_blend": float(args.scripted_action_prior_blend),
        "rm75_scripted_action_prior_approach_steps": int(args.scripted_action_prior_approach_steps),
        "rm75_scripted_action_prior_close_steps": int(args.scripted_action_prior_close_steps),
        "rm75_scripted_action_prior_hold_steps": int(args.scripted_action_prior_hold_steps),
        "rm75_scripted_action_prior_lift_steps": int(args.scripted_action_prior_lift_steps),
        "rm75_scripted_action_prior_lift_height": float(args.scripted_action_prior_lift_height),
        "rm75_scripted_action_prior_pos_gain": float(args.scripted_action_prior_pos_gain),
        "rm75_scripted_action_prior_mode": str(args.scripted_mode).replace("approach_", ""),
        "rm75_scripted_action_prior_target_max_xy_step": float(args.target_track_max_xy_step),
        "rm75_scripted_action_prior_target_max_z_step": float(args.target_track_max_z_step),
        "rm75_scripted_action_prior_target_gain": float(args.target_track_gain),
        "rm75_scripted_action_prior_target_z_deadband": float(args.target_track_z_deadband),
        "rm75_scripted_action_prior_target_pre_lift_height": 0.0,
        "rm75_scripted_action_prior_target_require_stable": bool(args.target_track_require_stable),
        "rm75_scripted_action_prior_thumb_yaw": float(args.scripted_action_prior_thumb_yaw),
        "rm75_scripted_action_prior_thumb_pitch": float(args.scripted_action_prior_thumb_pitch),
        "rm75_scripted_action_prior_finger_close": float(args.scripted_action_prior_finger_close),
        "rm75_scripted_action_prior_pinky_close": float(args.scripted_action_prior_pinky_close),
        "rm75_post_stable_lift_only": bool(args.post_stable_lift_only),
        "rm75_post_stable_max_xy_speed": float(args.post_stable_max_xy_speed),
        "rm75_post_stable_max_down_speed": float(args.post_stable_max_down_speed),
        "rm75_post_stable_max_up_speed": float(args.post_stable_max_up_speed),
        "rm75_post_stable_lift_bias": float(args.post_stable_lift_bias),
        "rm75_post_stable_xy_drift_correction_gain": float(args.post_stable_xy_drift_correction_gain),
        "rm75_post_stable_xy_drift_correction_max_speed": float(
            args.post_stable_xy_drift_correction_max_speed
        ),
        "rm75_post_contact_lift_assist": bool(args.post_contact_lift_assist),
        "rm75_post_contact_lift_min_hold_steps": int(args.post_contact_min_hold_steps),
        "rm75_post_contact_lift_min_non_thumb_contacts": int(args.post_contact_min_non_thumb_contacts),
        "rm75_post_contact_max_xy_speed": float(args.post_contact_max_xy_speed),
        "rm75_post_contact_max_down_speed": float(args.post_contact_max_down_speed),
        "rm75_post_contact_max_up_speed": float(args.post_contact_max_up_speed),
        "rm75_post_contact_lift_bias": float(args.post_contact_lift_bias),
        "rm75_post_contact_xy_drift_correction_gain": float(args.post_contact_xy_drift_correction_gain),
        "rm75_post_contact_xy_drift_correction_max_speed": float(
            args.post_contact_xy_drift_correction_max_speed
        ),
        "reward_kwargs": {
            "obj_com_done_thresh": 0.35,
            "no_contact_grace_steps": int(args.no_contact_grace_steps),
            "bad_push_done": bool(args.bad_push_done),
            "bad_push_min_step": 6,
            "bad_push_drift_thresh": 0.08,
            "bad_push_tilt_thresh": 0.75,
            "required_non_thumb_contacts": int(args.required_non_thumb_contacts),
        },
    }


def _scripted_action(env, args: argparse.Namespace) -> np.ndarray:
    if args.scripted_mode == "approach_target":
        return _approach_target_action(
            env,
            args.close_val,
            np.asarray(args.approach_delta, dtype=np.float32),
            args.approach_steps,
            args.close_steps,
            args.hold_steps,
            args.lift_steps,
            thumb_yaw=args.thumb_yaw,
            thumb_pitch=args.thumb_pitch,
            finger_close=args.finger_close,
            pinky_close=args.pinky_close,
            angular_action=_scheduled_angular_action(env, args),
            max_xy_step=args.target_track_max_xy_step,
            max_z_step=args.target_track_max_z_step,
            target_gain=args.target_track_gain,
            z_deadband=args.target_track_z_deadband,
            start_step=args.target_track_start_step,
            require_stable=args.target_track_require_stable,
        )
    return _approach_lift_action(
        env,
        args.close_val,
        np.asarray(args.approach_delta, dtype=np.float32),
        args.approach_steps,
        args.close_steps,
        args.hold_steps,
        args.lift_steps,
        args.lift_height,
        thumb_yaw=args.thumb_yaw,
        thumb_pitch=args.thumb_pitch,
        finger_close=args.finger_close,
        pinky_close=args.pinky_close,
        angular_action=_scheduled_angular_action(env, args),
    )


def _scheduled_angular_action(env, args: argparse.Namespace) -> np.ndarray:
    post_step = int(env.current_step) - int(env.pregrasp_steps)
    if post_step < int(args.angular_start_step) or post_step > int(args.angular_end_step):
        return np.zeros(3, dtype=np.float32)
    return np.clip(np.asarray(args.angular_action, dtype=np.float32), -1.0, 1.0)


def _settle_object(env, steps: int) -> None:
    if steps <= 0:
        return
    target_qpos = np.asarray(env.robot.get_qpos(), dtype=np.float32).copy()
    env.robot.set_drive_target(target_qpos)
    env.robot.set_drive_velocity_target(np.zeros_like(target_qpos))
    for _ in range(int(steps)):
        if hasattr(env.manipulated_object, "set_velocity"):
            env.manipulated_object.set_velocity(np.zeros(3, dtype=np.float32))
        if hasattr(env.manipulated_object, "set_angular_velocity"):
            env.manipulated_object.set_angular_velocity(np.zeros(3, dtype=np.float32))
        env.robot.set_qf(env.robot.compute_passive_force(external=False, coriolis_and_centrifugal=False))
        env.scene.step()


@torch.no_grad()
def _policy_action(policy: FPOStatePolicy, obs: np.ndarray, device: torch.device) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
    return policy.act(obs_tensor, deterministic=True).detach().cpu().numpy()[0]


def _row(env, step: int, action: np.ndarray, expert: np.ndarray, reward: float, done: bool, info: dict) -> dict:
    object_pos = np.asarray(env.manipulated_object.get_pose().p, dtype=np.float32)
    target_pos = np.asarray(env.target_object.get_pose().p, dtype=np.float32)
    palm_pos = np.asarray(getattr(env, "palm_link", env.ee_link).get_pose().p, dtype=np.float32)
    object_target_delta = target_pos - object_pos
    palm_object_delta = palm_pos - object_pos
    row = {
        "step": int(step),
        "reward": float(reward),
        "done": bool(done),
        "object_x": float(object_pos[0]),
        "object_y": float(object_pos[1]),
        "object_z": float(object_pos[2]),
        "target_x": float(target_pos[0]),
        "target_y": float(target_pos[1]),
        "target_z": float(target_pos[2]),
        "obj_target_dx": float(object_target_delta[0]),
        "obj_target_dy": float(object_target_delta[1]),
        "obj_target_dz": float(object_target_delta[2]),
        "obj_target_dist": float(np.linalg.norm(object_target_delta)),
        "obj_z_over_target": float(max(float(object_pos[2] - target_pos[2]), 0.0)),
        "palm_x": float(palm_pos[0]),
        "palm_y": float(palm_pos[1]),
        "palm_z": float(palm_pos[2]),
        "palm_object_dx": float(palm_object_delta[0]),
        "palm_object_dy": float(palm_object_delta[1]),
        "palm_object_dz": float(palm_object_delta[2]),
        "palm_object_dist": float(np.linalg.norm(palm_object_delta)),
        "palm_obj_dist": float(info.get("palm_obj_dist", np.nan)),
        "min_fingertip_obj_dist": float(info.get("min_fingertip_obj_dist", np.nan)),
        "fingertip_obj_dist": float(info.get("fingertip_obj_dist", np.nan)),
        "contact_count": float(info.get("contact_count", 0.0)),
        "thumb_contact": bool(info.get("thumb_contact", False)),
        "pinky_contact": bool(info.get("pinky_contact", False)),
        "ignore_pinky_contact": bool(info.get("ignore_pinky_contact", False)),
        "non_thumb_contact_count": int(info.get("non_thumb_contact_count", 0)),
        "stable_grasp_contact": bool(info.get("stable_grasp_contact", False)),
        "obj_lift": float(info.get("obj_lift", 0.0)),
        "object_xy_drift": float(info.get("object_xy_drift", 0.0)),
        "object_xy_drift_x": float(info.get("object_xy_drift_x", 0.0)),
        "object_xy_drift_y": float(info.get("object_xy_drift_y", 0.0)),
        "object_tilt_err": float(info.get("object_tilt_err", 0.0)),
        "hand_close_fraction": float(info.get("hand_close_fraction", 0.0)),
        "hand_thumb_close_fraction": float(info.get("hand_thumb_close_fraction", 0.0)),
        "hand_main_close_fraction": float(info.get("hand_main_close_fraction", 0.0)),
        "hand_pinky_close_fraction": float(info.get("hand_pinky_close_fraction", 0.0)),
        "reference_close_fraction": float(info.get("reference_close_fraction", 0.0)),
        "pregrasp_success": bool(info.get("pregrasp_success", False)),
        "obj_com_err": float(info.get("obj_com_err", 0.0)),
        "contact_success": bool(info.get("contact_success", False)),
        "lift_success_5cm": bool(info.get("lift_success_5cm", False)),
        "lift_success_target": bool(info.get("lift_success_target", False)),
        "norm_success_3": bool(info.get("norm_success_3", False)),
        "norm_success_10": bool(info.get("norm_success_10", False)),
        "done_on_norm_success_10_active": bool(info.get("done_on_norm_success_10_active", False)),
        "post_stable_lift_only_active": bool(info.get("post_stable_lift_only_active", False)),
        "post_contact_lift_assist_active": bool(info.get("post_contact_lift_assist_active", False)),
    }
    diff = np.asarray(action, dtype=np.float32) - np.asarray(expert, dtype=np.float32)
    row["action_l2_vs_expert"] = float(np.linalg.norm(diff))
    for i, value in enumerate(np.asarray(action[:6], dtype=np.float32)):
        row[f"action_arm_{i}"] = float(value)
    for i, value in enumerate(np.asarray(expert[:6], dtype=np.float32)):
        row[f"expert_arm_{i}"] = float(value)
    active_action_indices = [6, 7, 10, 12, 14, 16]
    for i, index in enumerate(active_action_indices):
        if index < len(action):
            row[f"action_hand_{i}"] = float(action[index])
        if index < len(expert):
            row[f"expert_hand_{i}"] = float(expert[index])
    return row


def main() -> None:
    args = _parse_args()
    if args.video_path is None:
        os.environ.setdefault("VIVIDEX_HEADLESS_NO_RENDER", "1")
    else:
        os.environ.pop("VIVIDEX_HEADLESS_NO_RENDER", None)
    os.environ.setdefault("VIVIDEX_RM75_NATIVE_HAND_CONTROL", "1")
    os.environ.setdefault("VIVIDEX_RM75_HAND_DOF", "12")

    from hand_imitation.env.create_rm75_env import create_rm75_env

    device = _device(args.device)
    policy = _load_policy(args.checkpoint, device)
    env = create_rm75_env(
        args.seq_name,
        use_gui=False,
        is_eval=bool(args.is_eval or args.video_path is not None),
        is_vision=False,
        norm_traj=True,
        robot_name="rm75_inspire_right",
        task_kwargs=_task_kwargs(args),
    )
    env._stage = int(args.stage)
    if not bool(args.skip_explicit_approach_apply):
        apply_approach_pregrasp_reference(env, np.asarray(args.approach_delta, dtype=np.float32))
    rows: list[dict] = []
    frames: list[np.ndarray] = []
    video_env = GymWrapper(env) if args.video_path is not None else None
    try:
        obs = env.reset(seed=args.seed)
        _settle_object(env, max(int(args.settle_steps), 0))
        if args.settle_steps > 0:
            obs = env.get_observation()
        if video_env is not None:
            frames.append(video_env.render(mode="rgb_array"))
        for step in range(args.steps):
            expert = _scripted_action(env, args)
            if args.rollout == "scripted":
                action = expert.copy()
            else:
                action = _policy_action(policy, obs, device)
            obs, reward, done, info = env.step(action)
            rows.append(_row(env, step, action, expert, reward, done, info))
            if video_env is not None:
                frames.append(video_env.render(mode="rgb_array"))
            if done:
                break
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.video_path is not None:
        args.video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video_path, fps=max(int(args.fps), 1))
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    summary = {
        "rollout": args.rollout,
        "scripted_mode": args.scripted_mode,
        "steps": len(rows),
        "any_contact_steps": int(sum(float(row["contact_count"]) > 0.0 for row in rows)),
        "non_thumb_steps": int(sum(int(row["non_thumb_contact_count"]) > 0 for row in rows)),
        "stable_steps": int(sum(bool(row["stable_grasp_contact"]) for row in rows)),
        "contact_success_steps": int(sum(bool(row["contact_success"]) for row in rows)),
        "lift_success_steps": int(sum(bool(row["lift_success_5cm"]) for row in rows)),
        "lift_success_target_steps": int(sum(bool(row["lift_success_target"]) for row in rows)),
        "norm_success_steps": int(sum(bool(row["norm_success_10"]) for row in rows)),
        "pinky_contact_steps": int(sum(bool(row["pinky_contact"]) for row in rows)),
        "post_stable_steps": int(sum(bool(row["post_stable_lift_only_active"]) for row in rows)),
        "post_contact_assist_steps": int(sum(bool(row["post_contact_lift_assist_active"]) for row in rows)),
        "max_lift": max((float(row["obj_lift"]) for row in rows), default=0.0),
        "max_xy_drift": max((float(row["object_xy_drift"]) for row in rows), default=0.0),
        "max_obj_z_over_target": max((float(row["obj_z_over_target"]) for row in rows), default=0.0),
        "final_obj_tgt_dist": float(rows[-1]["obj_target_dist"]) if rows else None,
        "final": rows[-1] if rows else {},
    }
    stable_rows = [row for row in rows if bool(row["stable_grasp_contact"])]
    if len(stable_rows) >= 2:
        first = np.array(
            [
                float(stable_rows[0]["palm_object_dx"]),
                float(stable_rows[0]["palm_object_dy"]),
                float(stable_rows[0]["palm_object_dz"]),
            ],
            dtype=np.float32,
        )
        slips = []
        for row in stable_rows[1:]:
            rel = np.array(
                [
                    float(row["palm_object_dx"]),
                    float(row["palm_object_dy"]),
                    float(row["palm_object_dz"]),
                ],
                dtype=np.float32,
            )
            slips.append(float(np.linalg.norm(rel - first)))
        summary["stable_palm_object_slip"] = float(max(slips, default=0.0))
    else:
        summary["stable_palm_object_slip"] = 0.0
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
