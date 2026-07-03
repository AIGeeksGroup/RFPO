#!/usr/bin/env python3
"""Run a scripted RM75/RH56 grasp smoke test in the relocate environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "hand_imitation"))

import numpy as np

from hand_imitation.env import task_setting
from hand_imitation.env.motion_paths import resolve_existing_motion_path
from hand_imitation.env.rl_env.reference_utils import expand_rm75_active_hand_action, rm75_hand_qpos_for_qlimits
from hand_imitation.env.sim_env.constructor import add_default_scene_light
from tools.retarget_rm75_inspire_reference import retarget_motion_dict


DEFAULT_SEQ = "ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=None)
    parser.add_argument("--seq-name", default=DEFAULT_SEQ)
    parser.add_argument("--robot-name", default="rm75_inspire_right")
    parser.add_argument("--norm-traj", action="store_true")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--close-val", type=float, default=0.9)
    parser.add_argument("--thumb-yaw", type=float, default=0.90)
    parser.add_argument("--thumb-pitch", type=float, default=0.85)
    parser.add_argument("--finger-close", type=float, default=0.75)
    parser.add_argument("--pinky-close", type=float, default=None)
    parser.add_argument("--thumb-delay-steps", type=int, default=0)
    parser.add_argument("--hand-action", nargs=6, type=float, default=None)
    parser.add_argument(
        "--mode",
        choices=["close", "track_reference", "lift_hold", "approach_lift", "approach_target"],
        default="track_reference",
    )
    parser.add_argument("--approach-delta", nargs=3, type=float, default=[0.01, -0.06, -0.02])
    parser.add_argument("--angular-action", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--angular-start-step", type=int, default=0)
    parser.add_argument("--angular-end-step", type=int, default=1000000)
    parser.add_argument("--approach-steps", type=int, default=14)
    parser.add_argument("--close-steps", type=int, default=30)
    parser.add_argument("--hold-steps", type=int, default=36)
    parser.add_argument("--lift-steps", type=int, default=140)
    parser.add_argument("--lift-height", type=float, default=0.14)
    parser.add_argument("--target-track-max-xy-step", type=float, default=0.006)
    parser.add_argument("--target-track-max-z-step", type=float, default=0.006)
    parser.add_argument("--target-track-gain", type=float, default=0.75)
    parser.add_argument("--target-track-z-deadband", type=float, default=0.002)
    parser.add_argument("--target-track-start-step", type=int, default=None)
    parser.add_argument("--target-track-require-stable", action="store_true")
    parser.add_argument("--keep-reference-pregrasp", action="store_true")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--trace-path", default=None)
    parser.add_argument("--palm-offset", nargs=3, type=float, default=None)
    parser.add_argument(
        "--retarget-mode",
        choices=["reachable", "object-frame", "vividex-aligned", "opposed-envelope", "native-object-only"],
        default="reachable",
    )
    parser.add_argument("--seed-pinky-qpos", type=float, default=None)
    parser.add_argument("--robot-base-offset", nargs=3, type=float, default=None)
    parser.add_argument("--robot-base-rpy", nargs=3, type=float, default=None)
    parser.add_argument("--robot-base-yaw", type=float, default=0.0)
    parser.add_argument("--object-scale", type=float, default=None)
    parser.add_argument("--native-hand-control", action="store_true")
    parser.add_argument("--enforce-native-mimic-qpos", action="store_true")
    parser.add_argument("--native-apply-template-approach-pregrasp", action="store_true")
    parser.add_argument("--pregrasp-success-mode", default=None)
    parser.add_argument("--pregrasp-palm-dist-thresh", type=float, default=None)
    parser.add_argument("--pregrasp-min-finger-dist-thresh", type=float, default=None)
    parser.add_argument("--no-done-on-pregrasp-failure", action="store_true")
    parser.add_argument("--obj-com-done-thresh", type=float, default=None)
    parser.add_argument("--no-contact-grace-steps", type=int, default=None)
    parser.add_argument("--bad-push-done", action="store_true")
    parser.add_argument("--required-non-thumb-contacts", type=int, default=None)
    parser.add_argument("--force-imitate-steps", type=int, default=None)
    parser.add_argument("--min-imitate-steps", type=int, default=None)
    parser.add_argument("--continue-after-done", action="store_true")
    parser.add_argument("--render-home-prefix-steps", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--global-seed", type=int, default=0)
    return parser.parse_args()


def _task_kwargs(args: argparse.Namespace) -> dict:
    task_kwargs = {"action": "relocate", "reward_kwargs": {}}
    if args.robot_base_offset is not None:
        task_kwargs["rm75_robot_base_offset"] = list(args.robot_base_offset)
    if args.robot_base_rpy is not None:
        task_kwargs["rm75_robot_base_rpy"] = list(args.robot_base_rpy)
    if abs(float(args.robot_base_yaw)) > 1e-8:
        task_kwargs["rm75_robot_base_yaw"] = float(args.robot_base_yaw)
    if args.object_scale is not None:
        task_kwargs["object_scale"] = float(args.object_scale)
    if args.native_hand_control:
        task_kwargs["rm75_native_hand_control"] = True
    if args.enforce_native_mimic_qpos:
        task_kwargs["rm75_enforce_native_mimic_qpos"] = True
    if args.native_apply_template_approach_pregrasp:
        task_kwargs["rm75_native_apply_template_approach_pregrasp"] = True
    if args.pregrasp_success_mode is not None:
        task_kwargs["rm75_pregrasp_success_mode"] = args.pregrasp_success_mode
    if args.pregrasp_palm_dist_thresh is not None:
        task_kwargs["rm75_pregrasp_palm_dist_thresh"] = float(args.pregrasp_palm_dist_thresh)
    if args.pregrasp_min_finger_dist_thresh is not None:
        task_kwargs["rm75_pregrasp_min_finger_dist_thresh"] = float(args.pregrasp_min_finger_dist_thresh)
    if args.no_done_on_pregrasp_failure:
        task_kwargs["rm75_done_on_pregrasp_failure"] = False
    if args.obj_com_done_thresh is not None:
        task_kwargs["reward_kwargs"]["obj_com_done_thresh"] = float(args.obj_com_done_thresh)
    if args.no_contact_grace_steps is not None:
        task_kwargs["reward_kwargs"]["no_contact_grace_steps"] = int(args.no_contact_grace_steps)
    if args.bad_push_done:
        task_kwargs["reward_kwargs"]["bad_push_done"] = True
    if args.required_non_thumb_contacts is not None:
        task_kwargs["reward_kwargs"]["required_non_thumb_contacts"] = int(args.required_non_thumb_contacts)
    if args.force_imitate_steps is not None:
        task_kwargs["rm75_force_imitate_steps"] = int(args.force_imitate_steps)
    if args.min_imitate_steps is not None:
        task_kwargs["rm75_min_imitate_steps"] = int(args.min_imitate_steps)
    return task_kwargs

def _action_from_qpos(env, target_qpos: np.ndarray, close_val: float) -> np.ndarray:
    qpos = env.robot.get_qpos()
    action = np.zeros(env.action_dim, dtype=np.float32)
    qlimits = env.robot.get_qlimits()

    palm_error = np.asarray(target_qpos[: env.arm_dof] - qpos[: env.arm_dof], dtype=np.float32)
    action[:6] = np.clip(palm_error[:6] / 0.05, -1.0, 1.0)
    _set_active_hand_action(action, float(close_val))
    if qlimits.shape[0] >= env.robot.dof:
        action[6:] = np.clip(action[6:], -1.0, 1.0)
    return action


def _closing_hand_action(
    close_val: float,
    thumb_yaw: float | None = None,
    thumb_pitch: float | None = None,
    finger_close: float | None = None,
    pinky_close: float | None = None,
    hand_action: np.ndarray | None = None,
) -> np.ndarray:
    if hand_action is not None:
        hand_action = np.asarray(hand_action, dtype=np.float32)
        if hand_action.shape != (6,):
            raise ValueError(f"Expected hand_action with shape (6,), got {hand_action.shape}")
        return np.clip(hand_action, -1.0, 1.0)

    close_val = float(close_val)
    finger_close = close_val if finger_close is None else float(finger_close)
    pinky_close = finger_close if pinky_close is None else float(pinky_close)
    thumb_yaw = close_val if thumb_yaw is None else float(thumb_yaw)
    thumb_pitch = close_val if thumb_pitch is None else float(thumb_pitch)
    return np.clip(
        np.array(
            [
                thumb_yaw,
                thumb_pitch,
                finger_close,
                finger_close,
                finger_close,
                pinky_close,
            ],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )


def _set_active_hand_action(action: np.ndarray, active_hand_action: np.ndarray | float) -> None:
    """Write 6 logical RM75 finger groups into either 6-DoF or native 12-DoF actions."""
    hand_action_dim = int(action.shape[0]) - 6
    if np.isscalar(active_hand_action):
        active = np.full(6, float(active_hand_action), dtype=np.float32)
    else:
        active = np.asarray(active_hand_action, dtype=np.float32)
    if active.shape != (6,):
        raise ValueError(f"Expected 6 RM75 hand groups, got {active.shape}")
    action[6:] = expand_rm75_active_hand_action(active, hand_action_dim)


def _reference_tracking_action(
    env,
    close_val: float,
    thumb_yaw: float | None = None,
    thumb_pitch: float | None = None,
    finger_close: float | None = None,
    pinky_close: float | None = None,
    hand_action: np.ndarray | None = None,
    angular_action: np.ndarray | None = None,
) -> np.ndarray:
    action = np.zeros(env.action_dim, dtype=np.float32)
    if env.current_step <= env.pregrasp_steps:
        target_palm = env.cur_reference_motion["robot_pregrasp_jpos"][-1, 0]
        closing_action = np.zeros(6, dtype=np.float32)
    else:
        ref_idx = min(env.current_step - env.pregrasp_steps, len(env.cur_reference_motion["robot_jpos"]) - 1)
        target_palm = env.cur_reference_motion["robot_jpos"][ref_idx, 0]
        closing_action = _closing_hand_action(
            close_val,
            thumb_yaw=thumb_yaw,
            thumb_pitch=thumb_pitch,
            finger_close=finger_close,
            pinky_close=pinky_close,
            hand_action=hand_action,
        )

    palm_error = np.asarray(target_palm - env.palm_link.get_pose().p, dtype=np.float32)
    action[:3] = np.clip(palm_error / 0.04, -1.0, 1.0)
    action[3:6] = 0.0 if angular_action is None else np.asarray(angular_action, dtype=np.float32)
    _set_active_hand_action(action, closing_action)
    return action


def _lift_hold_action(
    env,
    close_val: float,
    close_steps: int,
    hold_steps: int,
    lift_steps: int,
    lift_height: float,
    thumb_yaw: float | None = None,
    thumb_pitch: float | None = None,
    finger_close: float | None = None,
    pinky_close: float | None = None,
    thumb_delay_steps: int = 0,
    hand_action: np.ndarray | None = None,
    angular_action: np.ndarray | None = None,
) -> np.ndarray:
    action = np.zeros(env.action_dim, dtype=np.float32)
    if env.current_step <= env.pregrasp_steps:
        target_palm = env.cur_reference_motion["robot_pregrasp_jpos"][-1, 0]
        closing_action = np.zeros(6, dtype=np.float32)
    else:
        grasp_step = env.current_step - env.pregrasp_steps
        close_alpha = np.clip(grasp_step / max(1, int(close_steps)), 0.0, 1.0)
        thumb_alpha = np.clip(
            (grasp_step - max(0, int(thumb_delay_steps))) / max(1, int(close_steps)),
            0.0,
            1.0,
        )
        lift_start = int(close_steps) + int(hold_steps)
        lift_alpha = np.clip((grasp_step - lift_start) / max(1, int(lift_steps)), 0.0, 1.0)
        target_palm = np.asarray(env.cur_reference_motion["robot_jpos"][0, 0], dtype=np.float32).copy()
        target_palm[2] += float(lift_height) * float(lift_alpha)
        full_close = _closing_hand_action(
            close_val,
            thumb_yaw=thumb_yaw,
            thumb_pitch=thumb_pitch,
            finger_close=finger_close,
            pinky_close=pinky_close,
            hand_action=hand_action,
        )
        closing_action = close_alpha * full_close
        closing_action[:2] = thumb_alpha * full_close[:2]

    palm_error = np.asarray(target_palm - env.palm_link.get_pose().p, dtype=np.float32)
    action[:3] = np.clip(palm_error / 0.04, -1.0, 1.0)
    action[3:6] = 0.0 if angular_action is None else np.asarray(angular_action, dtype=np.float32)
    _set_active_hand_action(action, closing_action)
    return action


def _approach_lift_action(
    env,
    close_val: float,
    approach_delta: np.ndarray,
    approach_steps: int,
    close_steps: int,
    hold_steps: int,
    lift_steps: int,
    lift_height: float,
    thumb_yaw: float | None = None,
    thumb_pitch: float | None = None,
    finger_close: float | None = None,
    pinky_close: float | None = None,
    thumb_delay_steps: int = 0,
    hand_action: np.ndarray | None = None,
    angular_action: np.ndarray | None = None,
) -> np.ndarray:
    action = np.zeros(env.action_dim, dtype=np.float32)
    if env.current_step <= env.pregrasp_steps:
        target_palm = env.cur_reference_motion["robot_pregrasp_jpos"][-1, 0]
        closing_action = np.zeros(6, dtype=np.float32)
    else:
        grasp_step = env.current_step - env.pregrasp_steps
        approach_steps = max(1, int(approach_steps))
        close_steps = max(1, int(close_steps))
        hold_steps = max(0, int(hold_steps))
        lift_steps = max(1, int(lift_steps))

        grasp_palm = np.asarray(env.cur_reference_motion["robot_jpos"][0, 0], dtype=np.float32)
        approach_delta = np.asarray(approach_delta, dtype=np.float32)
        approach_alpha = np.clip(grasp_step / approach_steps, 0.0, 1.0)
        close_alpha = np.clip((grasp_step - approach_steps) / close_steps, 0.0, 1.0)
        thumb_alpha = np.clip(
            (grasp_step - approach_steps - max(0, int(thumb_delay_steps))) / close_steps,
            0.0,
            1.0,
        )
        lift_start = approach_steps + close_steps + hold_steps
        lift_alpha = np.clip((grasp_step - lift_start) / lift_steps, 0.0, 1.0)

        target_palm = grasp_palm + (1.0 - approach_alpha) * approach_delta
        target_palm = target_palm.copy()
        target_palm[2] += float(lift_height) * float(lift_alpha)
        full_close = _closing_hand_action(
            close_val,
            thumb_yaw=thumb_yaw,
            thumb_pitch=thumb_pitch,
            finger_close=finger_close,
            pinky_close=pinky_close,
            hand_action=hand_action,
        )
        closing_action = close_alpha * full_close
        closing_action[:2] = thumb_alpha * full_close[:2]

    palm_error = np.asarray(target_palm - env.palm_link.get_pose().p, dtype=np.float32)
    action[:3] = np.clip(palm_error / 0.04, -1.0, 1.0)
    action[3:6] = 0.0 if angular_action is None else np.asarray(angular_action, dtype=np.float32)
    _set_active_hand_action(action, closing_action)
    return action


def _approach_target_action(
    env,
    close_val: float,
    approach_delta: np.ndarray,
    approach_steps: int,
    close_steps: int,
    hold_steps: int,
    lift_steps: int,
    thumb_yaw: float | None = None,
    thumb_pitch: float | None = None,
    finger_close: float | None = None,
    pinky_close: float | None = None,
    thumb_delay_steps: int = 0,
    hand_action: np.ndarray | None = None,
    angular_action: np.ndarray | None = None,
    max_xy_step: float = 0.006,
    max_z_step: float = 0.006,
    target_gain: float = 0.75,
    z_deadband: float = 0.002,
    start_step: int | None = None,
    require_stable: bool = False,
) -> np.ndarray:
    action = np.zeros(env.action_dim, dtype=np.float32)
    if env.current_step <= env.pregrasp_steps:
        target_palm = env.cur_reference_motion["robot_pregrasp_jpos"][-1, 0]
        closing_action = np.zeros(6, dtype=np.float32)
    else:
        grasp_step = env.current_step - env.pregrasp_steps
        approach_steps = max(1, int(approach_steps))
        close_steps = max(1, int(close_steps))
        hold_steps = max(0, int(hold_steps))
        lift_steps = max(1, int(lift_steps))

        grasp_palm = np.asarray(env.cur_reference_motion["robot_jpos"][0, 0], dtype=np.float32)
        approach_delta = np.asarray(approach_delta, dtype=np.float32)
        approach_alpha = np.clip(grasp_step / approach_steps, 0.0, 1.0)
        close_alpha = np.clip((grasp_step - approach_steps) / close_steps, 0.0, 1.0)
        thumb_alpha = np.clip(
            (grasp_step - approach_steps - max(0, int(thumb_delay_steps))) / close_steps,
            0.0,
            1.0,
        )
        lift_start = approach_steps + close_steps + hold_steps
        if start_step is not None:
            lift_start = int(start_step)

        target_palm = grasp_palm + (1.0 - approach_alpha) * approach_delta
        target_palm = target_palm.copy()
        full_close = _closing_hand_action(
            close_val,
            thumb_yaw=thumb_yaw,
            thumb_pitch=thumb_pitch,
            finger_close=finger_close,
            pinky_close=pinky_close,
            hand_action=hand_action,
        )
        closing_action = close_alpha * full_close
        closing_action[:2] = thumb_alpha * full_close[:2]

        stable_now = bool(getattr(env, "stable_grasp_contact", False))
        if stable_now:
            env.rm75_target_track_latched = True
        track_allowed = grasp_step >= lift_start and bool(getattr(env, "rm75_target_track_latched", False))
        if require_stable:
            track_allowed = track_allowed and stable_now

        if track_allowed:
            object_pos = np.asarray(env.manipulated_object.get_pose().p, dtype=np.float32)
            target_pos = np.asarray(env.target_object.get_pose().p, dtype=np.float32)
            object_error = target_pos - object_pos
            if abs(float(object_error[2])) < float(z_deadband):
                object_error[2] = 0.0
            step = np.asarray(object_error * float(target_gain), dtype=np.float32)
            max_xy_step = max(float(max_xy_step), 0.0)
            max_z_step = max(float(max_z_step), 0.0)
            step[:2] = np.clip(step[:2], -max_xy_step, max_xy_step)
            step[2] = np.clip(step[2], -max_z_step, max_z_step)
            target_palm = np.asarray(env.palm_link.get_pose().p, dtype=np.float32) + step
        else:
            lift_alpha = np.clip((grasp_step - lift_start) / lift_steps, 0.0, 1.0)
            target_palm[2] += min(float(getattr(env, "rm75_target_track_pre_lift_height", 0.0)), 0.02) * lift_alpha

    palm_error = np.asarray(target_palm - env.palm_link.get_pose().p, dtype=np.float32)
    action[:3] = np.clip(palm_error / 0.04, -1.0, 1.0)
    action[3:6] = 0.0 if angular_action is None else np.asarray(angular_action, dtype=np.float32)
    _set_active_hand_action(action, closing_action)
    return action


def _scheduled_angular_action(env, args: argparse.Namespace) -> np.ndarray:
    post_step = int(env.current_step) - int(env.pregrasp_steps)
    if post_step < int(args.angular_start_step) or post_step > int(args.angular_end_step):
        return np.zeros(3, dtype=np.float32)
    return np.clip(np.asarray(args.angular_action, dtype=np.float32), -1.0, 1.0)


def _scripted_target_palm(env, args: argparse.Namespace) -> np.ndarray:
    """Return the palm target used by the scripted controller before stepping."""
    if args.mode == "track_reference":
        if env.current_step <= env.pregrasp_steps:
            return np.asarray(env.cur_reference_motion["robot_pregrasp_jpos"][-1, 0], dtype=np.float32)
        ref_idx = min(env.current_step - env.pregrasp_steps, len(env.cur_reference_motion["robot_jpos"]) - 1)
        return np.asarray(env.cur_reference_motion["robot_jpos"][ref_idx, 0], dtype=np.float32)

    if args.mode == "lift_hold":
        if env.current_step <= env.pregrasp_steps:
            return np.asarray(env.cur_reference_motion["robot_pregrasp_jpos"][-1, 0], dtype=np.float32)
        grasp_step = env.current_step - env.pregrasp_steps
        lift_start = int(args.close_steps) + int(args.hold_steps)
        lift_alpha = np.clip((grasp_step - lift_start) / max(1, int(args.lift_steps)), 0.0, 1.0)
        target_palm = np.asarray(env.cur_reference_motion["robot_jpos"][0, 0], dtype=np.float32).copy()
        target_palm[2] += float(args.lift_height) * float(lift_alpha)
        return target_palm

    if args.mode == "approach_lift":
        if env.current_step <= env.pregrasp_steps:
            return np.asarray(env.cur_reference_motion["robot_pregrasp_jpos"][-1, 0], dtype=np.float32)
        grasp_step = env.current_step - env.pregrasp_steps
        approach_steps = max(1, int(args.approach_steps))
        close_steps = max(1, int(args.close_steps))
        hold_steps = max(0, int(args.hold_steps))
        lift_steps = max(1, int(args.lift_steps))
        grasp_palm = np.asarray(env.cur_reference_motion["robot_jpos"][0, 0], dtype=np.float32)
        approach_delta = np.asarray(args.approach_delta, dtype=np.float32)
        approach_alpha = np.clip(grasp_step / approach_steps, 0.0, 1.0)
        lift_start = approach_steps + close_steps + hold_steps
        lift_alpha = np.clip((grasp_step - lift_start) / lift_steps, 0.0, 1.0)
        target_palm = grasp_palm + (1.0 - approach_alpha) * approach_delta
        target_palm = target_palm.copy()
        target_palm[2] += float(args.lift_height) * float(lift_alpha)
        return target_palm

    if args.mode == "approach_target":
        if env.current_step <= env.pregrasp_steps:
            return np.asarray(env.cur_reference_motion["robot_pregrasp_jpos"][-1, 0], dtype=np.float32)
        object_pos = np.asarray(env.manipulated_object.get_pose().p, dtype=np.float32)
        target_pos = np.asarray(env.target_object.get_pose().p, dtype=np.float32)
        object_error = target_pos - object_pos
        if abs(float(object_error[2])) < float(args.target_track_z_deadband):
            object_error[2] = 0.0
        step = np.asarray(object_error * float(args.target_track_gain), dtype=np.float32)
        step[:2] = np.clip(step[:2], -float(args.target_track_max_xy_step), float(args.target_track_max_xy_step))
        step[2] = np.clip(step[2], -float(args.target_track_max_z_step), float(args.target_track_max_z_step))
        return np.asarray(env.palm_link.get_pose().p, dtype=np.float32) + step

    return np.asarray(env.palm_link.get_pose().p, dtype=np.float32)


def _contact_diagnostics(env) -> tuple[list[dict], np.ndarray, float]:
    """Return object-hand contact details with per-group impulse magnitudes."""
    contact_links = env.finger_contact_links + [env.palm_link]
    link_to_index = {link: idx for idx, link in enumerate(contact_links)}
    impulse_by_group = np.zeros(env.num_contact_groups, dtype=np.float32)
    object_pos = np.asarray(env.manipulated_object.get_pose().p, dtype=np.float32)
    details = []
    for contact in env.scene.get_contacts():
        actors = {contact.actor0, contact.actor1}
        if env.manipulated_object not in actors:
            continue
        other_actors = actors - {env.manipulated_object}
        if not other_actors:
            continue
        other_actor = other_actors.pop()
        if other_actor not in link_to_index:
            continue
        impulses = [np.asarray(point.impulse, dtype=np.float32) for point in contact.points]
        positions = [
            np.asarray(point.position, dtype=np.float32)
            for point in contact.points
            if hasattr(point, "position")
        ]
        normals = [
            np.asarray(point.normal, dtype=np.float32)
            for point in contact.points
            if hasattr(point, "normal")
        ]
        if impulses:
            impulse_arr = np.stack(impulses, axis=0)
            impulse_abs_sum = float(np.sum(np.abs(impulse_arr)))
            impulse_norm_sum = float(np.sum(np.linalg.norm(impulse_arr, axis=1)))
        else:
            impulse_abs_sum = 0.0
            impulse_norm_sum = 0.0
        mean_position = None
        radial_xy_unit = None
        radial_z = None
        if positions:
            mean_pos = np.mean(np.stack(positions, axis=0), axis=0)
            radial = mean_pos - object_pos
            radial_xy = radial[:2]
            radial_xy_norm = float(np.linalg.norm(radial_xy))
            mean_position = mean_pos.tolist()
            radial_xy_unit = (
                (radial_xy / max(radial_xy_norm, 1e-6)).astype(np.float32).tolist()
            )
            radial_z = float(radial[2])
        mean_normal = None
        if normals:
            mean_normal = np.mean(np.stack(normals, axis=0), axis=0).astype(np.float32).tolist()
        link_index = link_to_index[other_actor]
        group_id = int(env.finger_contact_ids[link_index])
        impulse_by_group[group_id] += impulse_abs_sum
        details.append(
            {
                "link": other_actor.get_name(),
                "group": group_id,
                "num_points": len(contact.points),
                "impulse_abs_sum": impulse_abs_sum,
                "impulse_norm_sum": impulse_norm_sum,
                "mean_position": mean_position,
                "object_radial_xy_unit": radial_xy_unit,
                "object_radial_z": radial_z,
                "mean_normal": mean_normal,
            }
        )
    return details, impulse_by_group, float(np.sum(impulse_by_group))


def _all_robot_object_contacts(env) -> list[dict]:
    """Return all robot-object contacts, including links outside the reward contact groups."""
    robot_links = set(env.robot.get_links())
    details = []
    for contact in env.scene.get_contacts():
        actors = {contact.actor0, contact.actor1}
        if env.manipulated_object not in actors:
            continue
        other_actors = actors - {env.manipulated_object}
        if not other_actors:
            continue
        other_actor = other_actors.pop()
        if other_actor not in robot_links:
            continue
        impulses = [np.asarray(point.impulse, dtype=np.float32) for point in contact.points]
        impulse_norm_sum = 0.0
        if impulses:
            impulse_arr = np.stack(impulses, axis=0)
            impulse_norm_sum = float(np.sum(np.linalg.norm(impulse_arr, axis=1)))
        details.append(
            {
                "link": other_actor.get_name(),
                "num_points": len(contact.points),
                "impulse_norm_sum": impulse_norm_sum,
            }
        )
    return details


def apply_approach_pregrasp_reference(env, approach_delta: np.ndarray) -> None:
    """Move the pregrasp palm target to the approach-side pose.

    RM75 reset solves IK to ``robot_pregrasp_jpos[-1, 0]``.  For an approach
    trajectory that pregrasp target must be outside the object; otherwise the
    hand can start at the final grasp pose and launch the object before the
    scripted approach has a chance to run.
    """
    approach_delta = np.asarray(approach_delta, dtype=np.float32)
    motion = getattr(env, "_reference_motion", None)
    if motion is None:
        motion = env.cur_reference_motion
    if "robot_pregrasp_jpos" not in motion or "robot_jpos" not in motion:
        return
    grasp_palm = np.asarray(motion["robot_jpos"][0, 0], dtype=np.float32)
    approach_palm = grasp_palm + approach_delta
    motion["rm75_template_approach_delta"] = approach_delta.copy()
    motion["robot_pregrasp_jpos"][-1, 0] = approach_palm
    if hasattr(env, "cur_reference_motion"):
        env.cur_reference_motion["rm75_template_approach_delta"] = approach_delta.copy()
    if "robot_pregrasp_jpos" in env.cur_reference_motion:
        env.cur_reference_motion["robot_pregrasp_jpos"][-1, 0] = approach_palm


def _rm75_home_qpos(env) -> np.ndarray:
    qpos = np.zeros(env.robot.dof, dtype=np.float32)
    qpos[: env.arm_dof] = np.asarray(env.robot_info.arm_init_qpos, dtype=np.float32)
    qpos[env.arm_dof :] = rm75_hand_qpos_for_qlimits(
        np.array([0.08, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32),
        env.robot.get_qlimits()[env.arm_dof :],
    )
    return qpos


def _render_home_prefix(env, video_env, frames: list, num_steps: int) -> None:
    if video_env is None or num_steps <= 0:
        return
    scene_state = env.scene.pack()
    pregrasp_qpos = np.asarray(env.robot.get_qpos(), dtype=np.float32).copy()
    pregrasp_target = np.asarray(env.robot.get_drive_target(), dtype=np.float32).copy()
    home_qpos = _rm75_home_qpos(env)

    for i in range(num_steps):
        alpha = i / max(1, num_steps - 1)
        qpos = (1.0 - alpha) * home_qpos + alpha * pregrasp_qpos
        env.robot.set_qpos(qpos)
        env.robot.set_drive_target(qpos)
        env.scene.step()
        frames.append(video_env.render(mode="rgb_array"))

    env.scene.unpack(scene_state)
    env.robot.set_qpos(pregrasp_qpos)
    env.robot.set_drive_target(pregrasp_target)
    env.scene.step()


def _settle_object(env, steps: int) -> None:
    if steps <= 0:
        return
    target_qpos = np.asarray(env.robot.get_qpos(), dtype=np.float32).copy()
    env.robot.set_drive_target(target_qpos)
    env.robot.set_drive_velocity_target(np.zeros_like(target_qpos))
    if hasattr(env.manipulated_object, "set_velocity"):
        env.manipulated_object.set_velocity(np.zeros(3, dtype=np.float32))
    if hasattr(env.manipulated_object, "set_angular_velocity"):
        env.manipulated_object.set_angular_velocity(np.zeros(3, dtype=np.float32))
    for _ in range(steps):
        env.robot.set_qf(env.robot.compute_passive_force(external=False, coriolis_and_centrifugal=False))
        env.scene.step()
        if hasattr(env.manipulated_object, "set_velocity"):
            env.manipulated_object.set_velocity(np.zeros(3, dtype=np.float32))
        if hasattr(env.manipulated_object, "set_angular_velocity"):
            env.manipulated_object.set_angular_velocity(np.zeros(3, dtype=np.float32))


def _longest_true_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _summarize_trace_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "alive_steps": 0,
            "contact_steps": 0,
            "stable_contact_steps": 0,
            "longest_stable_contact_run": 0,
            "max_stable_contact_hold_steps": 0,
            "thumb_contact_steps": 0,
            "non_thumb_contact_steps": 0,
            "max_contact_count": 0.0,
            "max_lift": 0.0,
            "final_lift": 0.0,
            "final_obj_com_err": 0.0,
            "final_object_xy_drift": 0.0,
            "final_object_tilt_err": 0.0,
            "final_rm75_task_score": 0.0,
        }

    stable_values = [bool(row.get("stable_grasp_contact", False)) for row in rows]
    return {
        "alive_steps": len(rows),
        "contact_steps": int(sum(float(row.get("contact_count", 0.0)) > 0.0 for row in rows)),
        "stable_contact_steps": int(sum(stable_values)),
        "longest_stable_contact_run": _longest_true_run(stable_values),
        "max_stable_contact_hold_steps": int(
            max(int(row.get("stable_contact_hold_steps", 0)) for row in rows)
        ),
        "thumb_contact_steps": int(sum(bool(row.get("thumb_contact", False)) for row in rows)),
        "non_thumb_contact_steps": int(
            sum(int(row.get("non_thumb_contact_count", 0)) > 0 for row in rows)
        ),
        "max_contact_count": float(max(float(row.get("contact_count", 0.0)) for row in rows)),
        "max_lift": float(max(float(row.get("obj_lift", 0.0)) for row in rows)),
        "final_lift": float(rows[-1].get("obj_lift", 0.0)),
        "final_obj_com_err": float(rows[-1].get("obj_com_err", 0.0)),
        "final_object_xy_drift": float(rows[-1].get("object_xy_drift", 0.0)),
        "final_object_tilt_err": float(rows[-1].get("object_tilt_err", 0.0)),
        "final_rm75_task_score": float(rows[-1].get("rm75_task_score", 0.0)),
    }


def main() -> None:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "hand_imitation"))

    args = parse_args()
    if args.native_hand_control:
        os.environ.setdefault("VIVIDEX_RM75_NATIVE_HAND_CONTROL", "1")
        os.environ.setdefault("VIVIDEX_RM75_HAND_DOF", "12")
        os.environ.setdefault(
            "VIVIDEX_RM75_URDF_OVERRIDE",
            str(repo_root / "assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_nocyl.urdf"),
        )
    if args.global_seed is not None:
        np.random.seed(args.global_seed)
    from hand_imitation.env.create_rm75_env import create_rm75_env
    from hand_imitation.env.rl_env.rm75_relocate_env import RM75RelocateRLEnv
    from hand_imitation.env.gym_wrapper import GymWrapper

    if args.palm_offset is None and args.src is None:
        env = create_rm75_env(
            args.seq_name,
            use_gui=False,
            is_eval=args.video_path is not None,
            is_vision=False,
            norm_traj=args.norm_traj,
            robot_name=args.robot_name,
            task_kwargs=_task_kwargs(args),
        )
    elif args.palm_offset is None:
        src_npz = np.load(args.src, allow_pickle=True)
        motion = {key: src_npz[key] for key in src_npz.files}
        motion["task_name"] = "relocate"
        renderer_kwargs = {}
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ.get("VIVIDEX_HEADLESS_NO_RENDER", "0") != "1":
            renderer_kwargs["device"] = "cuda"
        env = RM75RelocateRLEnv(
            motion_file=motion,
            use_gui=False,
            task_kwargs=_task_kwargs(args),
            is_eval=args.video_path is not None,
            is_vision=False,
            norm_traj=args.norm_traj,
            robot_name=args.robot_name,
            no_rgb=os.environ.get("VIVIDEX_HEADLESS_NO_RENDER", "0") == "1",
            need_offscreen_render=args.video_path is not None
            and os.environ.get("VIVIDEX_HEADLESS_NO_RENDER", "0") != "1",
            **renderer_kwargs,
        )
        if args.video_path is not None:
            env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
            add_default_scene_light(env.scene, env.renderer)
    else:
        src_path = args.src or resolve_existing_motion_path(repo_root, args.seq_name, args.norm_traj, args.robot_name)
        src_npz = np.load(src_path, allow_pickle=True)
        src = {key: src_npz[key] for key in src_npz.files}
        motion = retarget_motion_dict(
            src,
            palm_offset=np.asarray(args.palm_offset, dtype=np.float32),
            mode=args.retarget_mode,
            hand_dof=12 if args.native_hand_control else 6,
            native_mimic=bool(args.native_hand_control),
            seed_pinky_qpos=args.seed_pinky_qpos,
        )
        motion["task_name"] = "relocate"
        renderer_kwargs = {}
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ.get("VIVIDEX_HEADLESS_NO_RENDER", "0") != "1":
            renderer_kwargs["device"] = "cuda"
        env = RM75RelocateRLEnv(
            motion_file=motion,
            use_gui=False,
            is_eval=args.video_path is not None,
            is_vision=False,
            norm_traj=args.norm_traj,
            robot_name=args.robot_name,
            task_kwargs=_task_kwargs(args),
            no_rgb=args.video_path is None,
            need_offscreen_render=args.video_path is not None,
            **renderer_kwargs,
        )
        if args.video_path is not None:
            env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
            add_default_scene_light(env.scene, env.renderer)
    env._stage = args.stage
    if args.mode in ("approach_lift", "approach_target") and not args.keep_reference_pregrasp:
        apply_approach_pregrasp_reference(env, np.asarray(args.approach_delta, dtype=np.float32))
    if args.seed is None:
        env.reset()
    else:
        np.random.seed(args.seed)
        env.reset(seed=args.seed)
    _settle_object(env, args.settle_steps)
    video_env = GymWrapper(env) if args.video_path is not None else None
    frames = []
    if video_env is not None:
        _render_home_prefix(env, video_env, frames, args.render_home_prefix_steps)
        frames.append(video_env.render(mode="rgb_array"))
    trace = []

    print(
        "step,done,reward,pregrasp_success,contact_count,obj_lift,obj_com_err,"
        "hand_jpos_err,hand_mjpos_err,palm_obj_dist,min_tip_dist,no_contact_steps"
    )
    done = False
    best_lift = 0.0
    contact_steps = 0
    alive_steps = 0
    try:
        for _ in range(args.steps):
            target_palm_before_step = _scripted_target_palm(env, args)
            palm_before_step = np.asarray(env.palm_link.get_pose().p, dtype=np.float32).copy()
            ee_before_step = np.asarray(env.ee_link.get_pose().p, dtype=np.float32).copy()
            if args.mode == "track_reference":
                action = _reference_tracking_action(
                    env,
                    args.close_val,
                    thumb_yaw=args.thumb_yaw,
                    thumb_pitch=args.thumb_pitch,
                    finger_close=args.finger_close,
                    pinky_close=args.pinky_close,
                    thumb_delay_steps=args.thumb_delay_steps,
                    hand_action=np.asarray(args.hand_action, dtype=np.float32) if args.hand_action is not None else None,
                    angular_action=_scheduled_angular_action(env, args),
                )
            elif args.mode == "lift_hold":
                action = _lift_hold_action(
                    env,
                    args.close_val,
                    args.close_steps,
                    args.hold_steps,
                    args.lift_steps,
                    args.lift_height,
                    thumb_yaw=args.thumb_yaw,
                    thumb_pitch=args.thumb_pitch,
                    finger_close=args.finger_close,
                    pinky_close=args.pinky_close,
                    thumb_delay_steps=args.thumb_delay_steps,
                    hand_action=np.asarray(args.hand_action, dtype=np.float32) if args.hand_action is not None else None,
                    angular_action=_scheduled_angular_action(env, args),
                )
            elif args.mode == "approach_lift":
                action = _approach_lift_action(
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
                    thumb_delay_steps=args.thumb_delay_steps,
                    hand_action=np.asarray(args.hand_action, dtype=np.float32) if args.hand_action is not None else None,
                    angular_action=_scheduled_angular_action(env, args),
                )
            elif args.mode == "approach_target":
                action = _approach_target_action(
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
                    hand_action=np.asarray(args.hand_action, dtype=np.float32) if args.hand_action is not None else None,
                    angular_action=_scheduled_angular_action(env, args),
                    max_xy_step=args.target_track_max_xy_step,
                    max_z_step=args.target_track_max_z_step,
                    target_gain=args.target_track_gain,
                    z_deadband=args.target_track_z_deadband,
                    start_step=args.target_track_start_step,
                    require_stable=args.target_track_require_stable,
                )
            else:
                action = np.zeros(env.action_dim, dtype=np.float32)
                active_hand_action = (
                    _closing_hand_action(
                        args.close_val,
                        thumb_yaw=args.thumb_yaw,
                        thumb_pitch=args.thumb_pitch,
                        finger_close=args.finger_close,
                        pinky_close=args.pinky_close,
                        hand_action=np.asarray(args.hand_action, dtype=np.float32)
                        if args.hand_action is not None
                        else None,
                    )
                    if env.current_step > env.pregrasp_steps
                    else 0.0
                )
                _set_active_hand_action(action, active_hand_action)
            _, reward, done, info = env.step(action)
            contact_details, contact_impulse_by_group, contact_impulse_total = _contact_diagnostics(env)
            all_robot_object_contacts = _all_robot_object_contacts(env)
            if video_env is not None:
                frames.append(video_env.render(mode="rgb_array"))
            if args.trace_path is not None:
                trace.append(
                    {
                        "step": env.current_step,
                        "done": bool(done),
                        "reward": float(reward),
                        "object_pos": np.asarray(env.manipulated_object.get_pose().p, dtype=np.float32).copy(),
                        "target_pos": np.asarray(env.target_object.get_pose().p, dtype=np.float32).copy(),
                        "target_palm": np.asarray(target_palm_before_step, dtype=np.float32).copy(),
                        "palm_before_step": palm_before_step,
                        "ee_before_step": ee_before_step,
                        "palm_pos": np.asarray(env.palm_link.get_pose().p, dtype=np.float32).copy(),
                        "ee_pos": np.asarray(env.ee_link.get_pose().p, dtype=np.float32).copy(),
                        "tip_pos": np.asarray([link.get_pose().p for link in env.finger_tip_links], dtype=np.float32).copy(),
                        "hand_qpos": np.asarray(env.robot.get_qpos()[env.arm_dof:], dtype=np.float32).copy(),
                        "action": np.asarray(action, dtype=np.float32).copy(),
                        "contact": np.asarray(env.robot_object_contact, dtype=np.float32).copy(),
                        "contact_impulse_by_group": contact_impulse_by_group.copy(),
                        "contact_impulse_total": contact_impulse_total,
                        "contact_details": contact_details,
                        "all_robot_object_contacts": all_robot_object_contacts,
                        "contact_count": float(info.get("contact_count", 0.0)),
                        "stable_grasp_contact": bool(info.get("stable_grasp_contact", False)),
                        "stable_contact_hold_steps": int(info.get("stable_contact_hold_steps", 0)),
                        "thumb_contact": bool(info.get("thumb_contact", False)),
                        "non_thumb_contact_count": int(info.get("non_thumb_contact_count", 0)),
                        "obj_lift": float(info.get("obj_lift", 0.0)),
                        "obj_com_err": float(info.get("obj_com_err", 0.0)),
                        "object_xy_drift": float(info.get("object_xy_drift", 0.0)),
                        "object_tilt_err": float(info.get("object_tilt_err", 0.0)),
                        "rm75_task_score": float(info.get("rm75_task_score", 0.0)),
                        "control_error": float(info.get("control_error", 0.0)),
                    }
                )
            contact_count = float(info.get("contact_count", 0.0))
            best_lift = max(best_lift, float(info.get("obj_lift", 0.0)))
            contact_steps += int(contact_count > 0.0)
            alive_steps += 1
            print(
                f"{env.current_step},{int(done)},{float(reward):.6f},{int(bool(info['pregrasp_success']))},"
                f"{contact_count:.1f},{float(info['obj_lift']):.6f},{float(info['obj_com_err']):.6f},"
                f"{float(info['hand_jpos_err']):.6f},{float(info['hand_mjpos_err']):.6f},"
                f"{float(info.get('palm_obj_dist', 0.0)):.6f},"
                f"{float(info.get('min_fingertip_obj_dist', 0.0)):.6f},"
                f"{int(getattr(env, 'no_contact_steps', 0))},impulse={contact_impulse_total:.5f},"
                f"links={[item['link'] for item in contact_details]},"
                f"all_robot_links={[item['link'] for item in all_robot_object_contacts]}"
            )
            if done and not args.continue_after_done:
                break
    finally:
        env.close()

    summary = _summarize_trace_rows(trace)
    if not trace:
        summary = {
            "alive_steps": alive_steps,
            "contact_steps": contact_steps,
            "stable_contact_steps": 0,
            "longest_stable_contact_run": 0,
            "max_stable_contact_hold_steps": 0,
            "thumb_contact_steps": 0,
            "non_thumb_contact_steps": 0,
            "max_contact_count": 0.0,
            "max_lift": best_lift,
            "final_lift": 0.0,
            "final_obj_com_err": 0.0,
            "final_object_xy_drift": 0.0,
            "final_object_tilt_err": 0.0,
            "final_rm75_task_score": 0.0,
        }
    print(
        "summary: "
        f"alive={summary['alive_steps']} contact_steps={summary['contact_steps']} "
        f"stable_contact_steps={summary['stable_contact_steps']} "
        f"longest_stable_contact_run={summary['longest_stable_contact_run']} "
        f"max_stable_hold={summary['max_stable_contact_hold_steps']} "
        f"thumb_steps={summary['thumb_contact_steps']} non_thumb_steps={summary['non_thumb_contact_steps']} "
        f"max_contact_count={summary['max_contact_count']:.1f} "
        f"max_lift={summary['max_lift']:.6f} final_lift={summary['final_lift']:.6f} "
        f"final_obj_com_err={summary['final_obj_com_err']:.6f} "
        f"final_xy_drift={summary['final_object_xy_drift']:.6f} "
        f"final_tilt={summary['final_object_tilt_err']:.6f} "
        f"final_task_score={summary['final_rm75_task_score']:.3f} "
        f"done={int(done)} mode={args.mode} close_val={args.close_val}"
    )
    if args.video_path is not None:
        import imageio

        video_path = Path(args.video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=args.fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"saved_video: {video_path}")
    if args.trace_path is not None:
        trace_path = Path(args.trace_path)
        if trace_path.suffix != ".npz":
            trace_path = trace_path.with_suffix(trace_path.suffix + ".npz")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path,
            step=np.asarray([row["step"] for row in trace], dtype=np.int32),
            done=np.asarray([row["done"] for row in trace], dtype=np.bool_),
            reward=np.asarray([row["reward"] for row in trace], dtype=np.float32),
            object_pos=np.asarray([row["object_pos"] for row in trace], dtype=np.float32),
            target_pos=np.asarray([row["target_pos"] for row in trace], dtype=np.float32),
            target_palm=np.asarray([row["target_palm"] for row in trace], dtype=np.float32),
            palm_before_step=np.asarray([row["palm_before_step"] for row in trace], dtype=np.float32),
            ee_before_step=np.asarray([row["ee_before_step"] for row in trace], dtype=np.float32),
            palm_pos=np.asarray([row["palm_pos"] for row in trace], dtype=np.float32),
            ee_pos=np.asarray([row["ee_pos"] for row in trace], dtype=np.float32),
            tip_pos=np.asarray([row["tip_pos"] for row in trace], dtype=np.float32),
            hand_qpos=np.asarray([row["hand_qpos"] for row in trace], dtype=np.float32),
            action=np.asarray([row["action"] for row in trace], dtype=np.float32),
            contact=np.asarray([row["contact"] for row in trace], dtype=np.float32),
            contact_impulse_by_group=np.asarray([row["contact_impulse_by_group"] for row in trace], dtype=np.float32),
            contact_impulse_total=np.asarray([row["contact_impulse_total"] for row in trace], dtype=np.float32),
            contact_count=np.asarray([row["contact_count"] for row in trace], dtype=np.float32),
            stable_grasp_contact=np.asarray([row["stable_grasp_contact"] for row in trace], dtype=np.bool_),
            stable_contact_hold_steps=np.asarray(
                [row["stable_contact_hold_steps"] for row in trace], dtype=np.int32
            ),
            thumb_contact=np.asarray([row["thumb_contact"] for row in trace], dtype=np.bool_),
            non_thumb_contact_count=np.asarray(
                [row["non_thumb_contact_count"] for row in trace], dtype=np.int32
            ),
            obj_lift=np.asarray([row["obj_lift"] for row in trace], dtype=np.float32),
            obj_com_err=np.asarray([row["obj_com_err"] for row in trace], dtype=np.float32),
            object_xy_drift=np.asarray([row["object_xy_drift"] for row in trace], dtype=np.float32),
            object_tilt_err=np.asarray([row["object_tilt_err"] for row in trace], dtype=np.float32),
            rm75_task_score=np.asarray([row["rm75_task_score"] for row in trace], dtype=np.float32),
            control_error=np.asarray([row["control_error"] for row in trace], dtype=np.float32),
        )
        summary_path = trace_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        details_path = trace_path.with_suffix(".contacts.json")
        details_path.write_text(
            json.dumps(
                [
                    {
                        "step": int(row["step"]),
                        "contact_details": row["contact_details"],
                        "all_robot_object_contacts": row["all_robot_object_contacts"],
                    }
                    for row in trace
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved_trace: {trace_path}")
        print(f"saved_summary: {summary_path}")
        print(f"saved_contact_details: {details_path}")


if __name__ == "__main__":
    main()
