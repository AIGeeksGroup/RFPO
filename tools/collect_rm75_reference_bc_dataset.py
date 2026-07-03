#!/usr/bin/env python3
"""Collect RM75/RH56 reference-action demonstrations for Flow-BC.

This intentionally does not use a PPO checkpoint.  The dataset comes from the
retargeted ViViDex reference plus a scripted RM75 grasp controller, matching the
FPO++ recipe of Flow-BC initialization before online RFPO fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import _init_paths
from tqdm import tqdm

from tools.scripted_rm75_grasp_smoke import (
    DEFAULT_SEQ,
    _approach_lift_action,
    _approach_target_action,
    _lift_hold_action,
    _reference_tracking_action,
    apply_approach_pregrasp_reference,
)


@dataclass(frozen=True)
class ScriptedRM75BCConfig:
    seq_name: str = DEFAULT_SEQ
    robot_name: str = "rm75_inspire_right"
    norm_traj: bool = True
    stage: int = 0
    mode: str = "approach_lift"
    native_hand_control: bool = True
    enforce_native_mimic_qpos: bool = False
    native_apply_template_approach_pregrasp: bool = True
    robot_base_offset_x: float = 0.0
    robot_base_offset_y: float = 0.0
    robot_base_offset_z: float = 0.0
    robot_base_rpy_x: float = 0.0
    robot_base_rpy_y: float = 0.0
    robot_base_rpy_z: float = 0.0
    robot_base_yaw: float = 0.0
    pregrasp_success_mode: str = "proximity"
    pregrasp_palm_dist_thresh: float = 0.22
    pregrasp_min_finger_dist_thresh: float = 0.10
    done_on_pregrasp_failure: bool = True
    object_scale: float = 1.0
    force_imitate_steps: int = 205
    min_imitate_steps: int = 0
    done_on_norm_success_10: bool = True
    norm_success_lift_thresh: float = 0.08
    score_lift_target: float = 0.08
    num_trajs: int = 256
    max_attempts: int = 4096
    keep_failed: bool = True
    filter_success: bool = False
    min_reward: float | None = None
    min_contact_steps: int = 1
    min_stable_steps: int = 0
    min_thumb_steps: int = 0
    min_non_thumb_steps: int = 0
    min_lift: float = 0.0
    obj_com_done_thresh: float = 0.35
    no_contact_grace_steps: int = 60
    bad_push_done: bool = False
    required_non_thumb_contacts: int = 1
    close_val: float = 0.30
    thumb_yaw: float = 0.90
    thumb_pitch: float = 0.85
    finger_close: float = 0.75
    pinky_close: float | None = None
    thumb_delay_steps: int = 0
    pinky_action_value: float = 0.0
    approach_delta_x: float = 0.01
    approach_delta_y: float = -0.06
    approach_delta_z: float = -0.02
    angular_action_x: float = 0.0
    angular_action_y: float = 0.0
    angular_action_z: float = 0.0
    angular_start_step: int = 0
    angular_end_step: int = 0
    approach_steps: int = 14
    close_steps: int = 30
    hold_steps: int = 36
    lift_steps: int = 140
    lift_height: float = 0.14
    target_track_max_xy_step: float = 0.006
    target_track_max_z_step: float = 0.006
    target_track_gain: float = 0.75
    target_track_z_deadband: float = 0.002
    target_track_start_step: int | None = None
    target_track_require_stable: bool = False
    keep_reference_pregrasp: bool = False
    settle_steps: int = 8
    execute_arm_scale: float = 1.0
    execute_hand_scale: float = 1.0
    seed: int = 0


def _parse_args() -> argparse.Namespace:
    defaults = ScriptedRM75BCConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--seq-name", default=defaults.seq_name)
    parser.add_argument("--robot-name", default=defaults.robot_name)
    parser.add_argument("--norm-traj", action="store_true", default=defaults.norm_traj)
    parser.add_argument("--stage", type=int, default=defaults.stage)
    parser.add_argument(
        "--mode",
        choices=["track_reference", "lift_hold", "approach_lift", "approach_target"],
        default=defaults.mode,
    )
    parser.add_argument("--native-hand-control", action="store_true", default=defaults.native_hand_control)
    parser.add_argument("--enforce-native-mimic-qpos", action="store_true", default=defaults.enforce_native_mimic_qpos)
    parser.add_argument(
        "--native-apply-template-approach-pregrasp",
        action="store_true",
        default=defaults.native_apply_template_approach_pregrasp,
    )
    parser.add_argument("--robot-base-offset", nargs=3, type=float, default=[
        defaults.robot_base_offset_x,
        defaults.robot_base_offset_y,
        defaults.robot_base_offset_z,
    ])
    parser.add_argument("--robot-base-rpy", nargs=3, type=float, default=[
        defaults.robot_base_rpy_x,
        defaults.robot_base_rpy_y,
        defaults.robot_base_rpy_z,
    ])
    parser.add_argument("--robot-base-yaw", type=float, default=defaults.robot_base_yaw)
    parser.add_argument("--pregrasp-success-mode", default=defaults.pregrasp_success_mode)
    parser.add_argument("--pregrasp-palm-dist-thresh", type=float, default=defaults.pregrasp_palm_dist_thresh)
    parser.add_argument(
        "--pregrasp-min-finger-dist-thresh",
        type=float,
        default=defaults.pregrasp_min_finger_dist_thresh,
    )
    parser.add_argument("--no-done-on-pregrasp-failure", action="store_true")
    parser.add_argument("--object-scale", type=float, default=defaults.object_scale)
    parser.add_argument("--force-imitate-steps", type=int, default=defaults.force_imitate_steps)
    parser.add_argument("--min-imitate-steps", type=int, default=defaults.min_imitate_steps)
    parser.add_argument(
        "--done-on-norm-success-10",
        action="store_true",
        default=defaults.done_on_norm_success_10,
    )
    parser.add_argument(
        "--no-done-on-norm-success-10",
        action="store_false",
        dest="done_on_norm_success_10",
    )
    parser.add_argument("--norm-success-lift-thresh", type=float, default=defaults.norm_success_lift_thresh)
    parser.add_argument("--score-lift-target", type=float, default=defaults.score_lift_target)
    parser.add_argument("-n", "--num-trajs", type=int, default=defaults.num_trajs)
    parser.add_argument("--max-attempts", type=int, default=defaults.max_attempts)
    parser.add_argument("--keep-failed", action="store_true", default=defaults.keep_failed)
    parser.add_argument("--filter-success", action="store_true", help="Only keep episodes passing the contact/lift filters.")
    parser.add_argument("--min-reward", type=float, default=defaults.min_reward)
    parser.add_argument("--min-contact-steps", type=int, default=defaults.min_contact_steps)
    parser.add_argument("--min-stable-steps", type=int, default=defaults.min_stable_steps)
    parser.add_argument("--min-thumb-steps", type=int, default=defaults.min_thumb_steps)
    parser.add_argument("--min-non-thumb-steps", type=int, default=defaults.min_non_thumb_steps)
    parser.add_argument("--min-lift", type=float, default=defaults.min_lift)
    parser.add_argument("--obj-com-done-thresh", type=float, default=defaults.obj_com_done_thresh)
    parser.add_argument("--no-contact-grace-steps", type=int, default=defaults.no_contact_grace_steps)
    parser.add_argument("--bad-push-done", action="store_true", default=defaults.bad_push_done)
    parser.add_argument("--required-non-thumb-contacts", type=int, default=defaults.required_non_thumb_contacts)
    parser.add_argument("--close-val", type=float, default=defaults.close_val)
    parser.add_argument("--thumb-yaw", type=float, default=defaults.thumb_yaw)
    parser.add_argument("--thumb-pitch", type=float, default=defaults.thumb_pitch)
    parser.add_argument("--finger-close", type=float, default=defaults.finger_close)
    parser.add_argument("--pinky-close", type=float, default=defaults.pinky_close)
    parser.add_argument("--thumb-delay-steps", type=int, default=defaults.thumb_delay_steps)
    parser.add_argument("--pinky-action-value", type=float, default=defaults.pinky_action_value)
    parser.add_argument("--approach-delta", nargs=3, type=float, default=[
        defaults.approach_delta_x,
        defaults.approach_delta_y,
        defaults.approach_delta_z,
    ])
    parser.add_argument("--angular-action", nargs=3, type=float, default=[
        defaults.angular_action_x,
        defaults.angular_action_y,
        defaults.angular_action_z,
    ])
    parser.add_argument("--angular-start-step", type=int, default=defaults.angular_start_step)
    parser.add_argument("--angular-end-step", type=int, default=defaults.angular_end_step)
    parser.add_argument("--approach-steps", type=int, default=defaults.approach_steps)
    parser.add_argument("--close-steps", type=int, default=defaults.close_steps)
    parser.add_argument("--hold-steps", type=int, default=defaults.hold_steps)
    parser.add_argument("--lift-steps", type=int, default=defaults.lift_steps)
    parser.add_argument("--lift-height", type=float, default=defaults.lift_height)
    parser.add_argument("--target-track-max-xy-step", type=float, default=defaults.target_track_max_xy_step)
    parser.add_argument("--target-track-max-z-step", type=float, default=defaults.target_track_max_z_step)
    parser.add_argument("--target-track-gain", type=float, default=defaults.target_track_gain)
    parser.add_argument("--target-track-z-deadband", type=float, default=defaults.target_track_z_deadband)
    parser.add_argument("--target-track-start-step", type=int, default=defaults.target_track_start_step)
    parser.add_argument("--target-track-require-stable", action="store_true", default=defaults.target_track_require_stable)
    parser.add_argument("--keep-reference-pregrasp", action="store_true", default=defaults.keep_reference_pregrasp)
    parser.add_argument("--settle-steps", type=int, default=defaults.settle_steps)
    parser.add_argument("--execute-arm-scale", type=float, default=defaults.execute_arm_scale)
    parser.add_argument("--execute-hand-scale", type=float, default=defaults.execute_hand_scale)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args()


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


def _scheduled_angular_action(env, args: argparse.Namespace) -> np.ndarray:
    post_step = int(env.current_step) - int(env.pregrasp_steps)
    if post_step < int(args.angular_start_step) or post_step > int(args.angular_end_step):
        return np.zeros(3, dtype=np.float32)
    return np.clip(np.asarray(args.angular_action, dtype=np.float32), -1.0, 1.0)


def _scripted_action(env, args: argparse.Namespace) -> np.ndarray:
    angular_action = _scheduled_angular_action(env, args)
    if args.mode == "track_reference":
        return _reference_tracking_action(
            env,
            args.close_val,
            thumb_yaw=args.thumb_yaw,
            thumb_pitch=args.thumb_pitch,
            finger_close=args.finger_close,
            pinky_close=args.pinky_close,
            angular_action=angular_action,
        )
    if args.mode == "lift_hold":
        return _lift_hold_action(
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
            angular_action=angular_action,
        )
    if args.mode == "approach_target":
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
            thumb_delay_steps=args.thumb_delay_steps,
            angular_action=angular_action,
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
        thumb_delay_steps=args.thumb_delay_steps,
        angular_action=angular_action,
    )


def _executed_action(label_action: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    action = np.asarray(label_action, dtype=np.float32).copy()
    action[:6] *= float(args.execute_arm_scale)
    action[6:] *= float(args.execute_hand_scale)
    hand_action_dim = int(action.shape[0]) - 6
    pinky_value = float(np.clip(args.pinky_action_value, -1.0, 1.0))
    if hand_action_dim == 12:
        action[16:18] = pinky_value
    elif hand_action_dim == 6:
        action[11] = pinky_value
    return np.clip(action, -1.0, 1.0)


def _accept_episode(
    total_reward: float,
    contact_steps: int,
    stable_steps: int,
    thumb_steps: int,
    non_thumb_steps: int,
    best_lift: float,
    args: argparse.Namespace,
) -> bool:
    if args.min_reward is not None and total_reward < args.min_reward:
        return False
    if contact_steps < args.min_contact_steps:
        return False
    if stable_steps < args.min_stable_steps:
        return False
    if thumb_steps < args.min_thumb_steps:
        return False
    if non_thumb_steps < args.min_non_thumb_steps:
        return False
    if best_lift < args.min_lift:
        return False
    return True


def _task_kwargs(args: argparse.Namespace) -> dict:
    return {
        "action": "relocate",
        "rm75_native_hand_control": bool(args.native_hand_control),
        "rm75_enforce_native_mimic_qpos": bool(args.enforce_native_mimic_qpos),
        "rm75_native_apply_template_approach_pregrasp": bool(args.native_apply_template_approach_pregrasp),
        "rm75_robot_base_offset": list(args.robot_base_offset),
        "rm75_robot_base_rpy": list(args.robot_base_rpy),
        "rm75_robot_base_yaw": float(args.robot_base_yaw),
        "rm75_pregrasp_success_mode": str(args.pregrasp_success_mode),
        "rm75_pregrasp_palm_dist_thresh": float(args.pregrasp_palm_dist_thresh),
        "rm75_pregrasp_min_finger_dist_thresh": float(args.pregrasp_min_finger_dist_thresh),
        "rm75_done_on_pregrasp_failure": not bool(args.no_done_on_pregrasp_failure),
        "object_scale": float(args.object_scale),
        "rm75_force_imitate_steps": int(args.force_imitate_steps),
        "rm75_min_imitate_steps": int(args.min_imitate_steps),
        "rm75_done_on_norm_success_10": bool(args.done_on_norm_success_10),
        "rm75_norm_success_lift_thresh": float(args.norm_success_lift_thresh),
        "rm75_norm_success_requires_contact": True,
        "rm75_success_min_non_thumb_contacts": int(args.required_non_thumb_contacts),
        "rm75_score_lift_target": float(args.score_lift_target),
        "rm75_ignore_pinky_contact": True,
        "rm75_disable_pinky_action": True,
        "rm75_pinky_action_value": float(args.pinky_action_value),
        "reward_kwargs": {
            "obj_com_done_thresh": float(args.obj_com_done_thresh),
            "no_contact_grace_steps": int(args.no_contact_grace_steps),
            "bad_push_done": bool(args.bad_push_done),
            "required_non_thumb_contacts": int(args.required_non_thumb_contacts),
        },
    }


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)
    os.environ.setdefault("VIVIDEX_HEADLESS_NO_RENDER", "1")
    if args.native_hand_control:
        os.environ["VIVIDEX_RM75_NATIVE_HAND_CONTROL"] = "1"
        os.environ.setdefault("VIVIDEX_RM75_HAND_DOF", "12")
    if args.enforce_native_mimic_qpos:
        os.environ["VIVIDEX_RM75_ENFORCE_NATIVE_MIMIC"] = "1"

    from hand_imitation.env.create_rm75_env import create_rm75_env

    env = create_rm75_env(
        args.seq_name,
        use_gui=False,
        is_eval=False,
        is_vision=False,
        norm_traj=args.norm_traj,
        robot_name=args.robot_name,
        task_kwargs=_task_kwargs(args),
    )
    env._stage = int(args.stage)
    if args.mode in ("approach_lift", "approach_target") and not bool(args.keep_reference_pregrasp):
        apply_approach_pregrasp_reference(env, np.asarray(args.approach_delta, dtype=np.float32))

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    episode_ends: list[int] = []
    episode_attempts: list[int] = []
    episode_reward: list[float] = []
    episode_contact_steps: list[int] = []
    episode_stable_steps: list[int] = []
    episode_thumb_steps: list[int] = []
    episode_non_thumb_steps: list[int] = []
    episode_best_lift: list[float] = []
    accepted = 0
    attempts = 0

    try:
        with tqdm(total=args.num_trajs) as pbar:
            while accepted < args.num_trajs and attempts < args.max_attempts:
                attempts += 1
                ep_obs: list[np.ndarray] = []
                ep_actions: list[np.ndarray] = []
                ep_rewards: list[float] = []
                contact_steps = 0
                stable_steps = 0
                thumb_steps = 0
                non_thumb_steps = 0
                best_lift = 0.0
                obs = env.reset(seed=args.seed + attempts)
                _settle_object(env, args.settle_steps)
                obs = env.get_observation()

                for _ in range(env.horizon):
                    action = np.asarray(_scripted_action(env, args), dtype=np.float32)
                    executed_action = _executed_action(action, args)
                    ep_obs.append(np.asarray(obs, dtype=np.float32))
                    ep_actions.append(executed_action)
                    obs, reward, done, info = env.step(executed_action)
                    ep_rewards.append(float(reward))
                    contact_steps += int(float(info.get("contact_count", 0.0)) > 0.0)
                    stable_steps += int(bool(info.get("stable_grasp_contact", False)))
                    thumb_steps += int(bool(info.get("thumb_contact", False)))
                    non_thumb_steps += int(int(info.get("non_thumb_contact_count", 0)) > 0)
                    best_lift = max(best_lift, float(info.get("obj_lift", 0.0)))
                    if done:
                        break

                total_reward = float(np.sum(ep_rewards))
                accepted_by_filter = _accept_episode(
                    total_reward,
                    contact_steps,
                    stable_steps,
                    thumb_steps,
                    non_thumb_steps,
                    best_lift,
                    args,
                )
                keep = accepted_by_filter or (args.keep_failed and not args.filter_success)
                if keep:
                    observations.extend(ep_obs)
                    actions.extend(ep_actions)
                    rewards.extend(ep_rewards)
                    episode_ends.append(len(observations))
                    episode_attempts.append(attempts)
                    episode_reward.append(total_reward)
                    episode_contact_steps.append(contact_steps)
                    episode_stable_steps.append(stable_steps)
                    episode_thumb_steps.append(thumb_steps)
                    episode_non_thumb_steps.append(non_thumb_steps)
                    episode_best_lift.append(best_lift)
                    accepted += 1
                    pbar.update(1)
                pbar.set_description(
                    f"accepted {accepted}/{attempts} reward {total_reward:.2f} "
                    f"contact {contact_steps} stable {stable_steps} "
                    f"thumb {thumb_steps} nonthumb {non_thumb_steps} lift {best_lift:.3f}"
                )
    finally:
        env.close()

    if not observations:
        raise RuntimeError("No RM75 reference episodes were collected.")

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    cfg = ScriptedRM75BCConfig(
        seq_name=args.seq_name,
        robot_name=args.robot_name,
        norm_traj=bool(args.norm_traj),
        stage=int(args.stage),
        native_hand_control=bool(args.native_hand_control),
        enforce_native_mimic_qpos=bool(args.enforce_native_mimic_qpos),
        native_apply_template_approach_pregrasp=bool(args.native_apply_template_approach_pregrasp),
        robot_base_offset_x=float(args.robot_base_offset[0]),
        robot_base_offset_y=float(args.robot_base_offset[1]),
        robot_base_offset_z=float(args.robot_base_offset[2]),
        robot_base_rpy_x=float(args.robot_base_rpy[0]),
        robot_base_rpy_y=float(args.robot_base_rpy[1]),
        robot_base_rpy_z=float(args.robot_base_rpy[2]),
        robot_base_yaw=float(args.robot_base_yaw),
        pregrasp_success_mode=str(args.pregrasp_success_mode),
        pregrasp_palm_dist_thresh=float(args.pregrasp_palm_dist_thresh),
        pregrasp_min_finger_dist_thresh=float(args.pregrasp_min_finger_dist_thresh),
        done_on_pregrasp_failure=not bool(args.no_done_on_pregrasp_failure),
        object_scale=float(args.object_scale),
        force_imitate_steps=int(args.force_imitate_steps),
        min_imitate_steps=int(args.min_imitate_steps),
        done_on_norm_success_10=bool(args.done_on_norm_success_10),
        norm_success_lift_thresh=float(args.norm_success_lift_thresh),
        score_lift_target=float(args.score_lift_target),
        mode=args.mode,
        num_trajs=int(args.num_trajs),
        max_attempts=int(args.max_attempts),
        keep_failed=bool(args.keep_failed),
        filter_success=bool(args.filter_success),
        min_reward=args.min_reward,
        min_contact_steps=int(args.min_contact_steps),
        min_stable_steps=int(args.min_stable_steps),
        min_thumb_steps=int(args.min_thumb_steps),
        min_non_thumb_steps=int(args.min_non_thumb_steps),
        min_lift=float(args.min_lift),
        obj_com_done_thresh=float(args.obj_com_done_thresh),
        no_contact_grace_steps=int(args.no_contact_grace_steps),
        bad_push_done=bool(args.bad_push_done),
        required_non_thumb_contacts=int(args.required_non_thumb_contacts),
        close_val=float(args.close_val),
        thumb_yaw=float(args.thumb_yaw),
        thumb_pitch=float(args.thumb_pitch),
        finger_close=float(args.finger_close),
        pinky_close=None if args.pinky_close is None else float(args.pinky_close),
        thumb_delay_steps=int(args.thumb_delay_steps),
        pinky_action_value=float(args.pinky_action_value),
        approach_delta_x=float(args.approach_delta[0]),
        approach_delta_y=float(args.approach_delta[1]),
        approach_delta_z=float(args.approach_delta[2]),
        angular_action_x=float(args.angular_action[0]),
        angular_action_y=float(args.angular_action[1]),
        angular_action_z=float(args.angular_action[2]),
        angular_start_step=int(args.angular_start_step),
        angular_end_step=int(args.angular_end_step),
        approach_steps=int(args.approach_steps),
        close_steps=int(args.close_steps),
        hold_steps=int(args.hold_steps),
        lift_steps=int(args.lift_steps),
        lift_height=float(args.lift_height),
        target_track_max_xy_step=float(args.target_track_max_xy_step),
        target_track_max_z_step=float(args.target_track_max_z_step),
        target_track_gain=float(args.target_track_gain),
        target_track_z_deadband=float(args.target_track_z_deadband),
        target_track_start_step=args.target_track_start_step,
        target_track_require_stable=bool(args.target_track_require_stable),
        keep_reference_pregrasp=bool(args.keep_reference_pregrasp),
        settle_steps=int(args.settle_steps),
        execute_arm_scale=float(args.execute_arm_scale),
        execute_hand_scale=float(args.execute_hand_scale),
        seed=int(args.seed),
    )
    episode_reward_np = np.asarray(episode_reward, dtype=np.float32)
    episode_contact_steps_np = np.asarray(episode_contact_steps, dtype=np.int32)
    episode_stable_steps_np = np.asarray(episode_stable_steps, dtype=np.int32)
    episode_thumb_steps_np = np.asarray(episode_thumb_steps, dtype=np.int32)
    episode_non_thumb_steps_np = np.asarray(episode_non_thumb_steps, dtype=np.int32)
    episode_best_lift_np = np.asarray(episode_best_lift, dtype=np.float32)
    np.savez_compressed(
        output,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
        episode_attempts=np.asarray(episode_attempts, dtype=np.int32),
        episode_reward=episode_reward_np,
        episode_contact_steps=episode_contact_steps_np,
        episode_stable_steps=episode_stable_steps_np,
        episode_thumb_steps=episode_thumb_steps_np,
        episode_non_thumb_steps=episode_non_thumb_steps_np,
        episode_best_lift=episode_best_lift_np,
        attempts=np.asarray(attempts, dtype=np.int32),
        acceptance_rate=np.asarray(len(episode_ends) / max(attempts, 1), dtype=np.float32),
        env_name=np.asarray(args.seq_name),
        robot_name=np.asarray(args.robot_name),
        stage=np.asarray(args.stage),
        source=np.asarray("rm75_reference_scripted"),
        config_json=np.asarray(json.dumps(asdict(cfg), sort_keys=True)),
    )
    reward_q = np.quantile(episode_reward_np, [0.0, 0.25, 0.5, 0.75, 1.0])
    contact_q = np.quantile(episode_contact_steps_np, [0.0, 0.25, 0.5, 0.75, 1.0])
    stable_q = np.quantile(episode_stable_steps_np, [0.0, 0.25, 0.5, 0.75, 1.0])
    thumb_q = np.quantile(episode_thumb_steps_np, [0.0, 0.25, 0.5, 0.75, 1.0])
    non_thumb_q = np.quantile(episode_non_thumb_steps_np, [0.0, 0.25, 0.5, 0.75, 1.0])
    lift_q = np.quantile(episode_best_lift_np, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(
        f"saved {len(episode_ends)} episodes, {len(observations)} transitions to {output}; "
        f"attempts={attempts}, acceptance_rate={len(episode_ends) / max(attempts, 1):.3f}, "
        f"mean_reward={np.mean(episode_reward):.3f}, "
        f"mean_contact_steps={np.mean(episode_contact_steps):.2f}, "
        f"mean_stable_steps={np.mean(episode_stable_steps):.2f}, "
        f"mean_thumb_steps={np.mean(episode_thumb_steps):.2f}, "
        f"mean_non_thumb_steps={np.mean(episode_non_thumb_steps):.2f}, "
        f"mean_best_lift={np.mean(episode_best_lift):.4f}"
    )
    print(
        "episode quality quantiles "
        f"reward[min/p25/median/p75/max]={reward_q.tolist()} "
        f"contact={contact_q.tolist()} stable={stable_q.tolist()} "
        f"thumb={thumb_q.tolist()} nonthumb={non_thumb_q.tolist()} lift={lift_q.tolist()}"
    )


if __name__ == "__main__":
    main()
