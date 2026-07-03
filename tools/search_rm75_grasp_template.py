#!/usr/bin/env python3
"""Search RM75/RH56 object-frame grasp templates with scripted rollouts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "hand_imitation"))

import numpy as np

from hand_imitation.env.motion_paths import resolve_existing_motion_path
from tools.retarget_rm75_inspire_reference import retarget_motion_dict
from tools.scripted_rm75_grasp_smoke import (
    DEFAULT_SEQ,
    _approach_lift_action,
    _contact_diagnostics,
    _lift_hold_action,
    _reference_tracking_action,
    _settle_object,
    apply_approach_pregrasp_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=None)
    parser.add_argument("--seq-name", default=DEFAULT_SEQ)
    parser.add_argument("--robot-name", default="rm75_inspire_right")
    parser.add_argument("--norm-traj", action="store_true")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--mode", choices=["track_reference", "lift_hold", "approach_lift"], default="lift_hold")
    parser.add_argument(
        "--retarget-mode",
        choices=["reachable", "object-frame", "vividex-aligned", "opposed-envelope", "native-object-only"],
        default="reachable",
    )
    parser.add_argument("--approach-deltas", nargs="+", type=float, default=None)
    parser.add_argument("--approach-steps", type=int, default=10)
    parser.add_argument("--close-steps", type=int, default=18)
    parser.add_argument("--hold-steps", type=int, default=10)
    parser.add_argument("--lift-steps", type=int, default=35)
    parser.add_argument("--lift-height", type=float, default=0.08)
    parser.add_argument("--angular-actions", nargs="+", type=float, default=None)
    parser.add_argument("--angular-start-step", type=int, default=0)
    parser.add_argument("--angular-end-step", type=int, default=1000000)
    parser.add_argument("--close-vals", nargs="+", type=float, default=[0.8, 0.9, 1.0])
    parser.add_argument("--thumb-yaws", nargs="+", type=float, default=None)
    parser.add_argument("--thumb-pitches", nargs="+", type=float, default=None)
    parser.add_argument("--finger-closes", nargs="+", type=float, default=None)
    parser.add_argument("--pinky-closes", nargs="+", type=float, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--no-retarget", action="store_true", help="Search actions on the source trajectory without changing palm offsets.")
    parser.add_argument("--base-offset", nargs=3, type=float, default=[0.057, -0.089, 0.090])
    parser.add_argument("--seed-pinky-qpos", type=float, default=None)
    parser.add_argument("--robot-base-offset", nargs=3, type=float, default=None)
    parser.add_argument("--robot-base-offsets", nargs="+", type=float, default=None)
    parser.add_argument("--robot-base-rpy", nargs=3, type=float, default=None)
    parser.add_argument("--robot-base-yaws", nargs="+", type=float, default=[0.0])
    parser.add_argument("--object-scale", type=float, default=None)
    parser.add_argument("--native-hand-control", action="store_true")
    parser.add_argument("--enforce-native-mimic-qpos", action="store_true")
    parser.add_argument("--native-apply-template-approach-pregrasp", action="store_true")
    parser.add_argument(
        "--apply-scripted-approach-pregrasp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also move robot_pregrasp_jpos to the searched approach pose before reset.",
    )
    parser.add_argument("--pregrasp-success-mode", default=None)
    parser.add_argument("--pregrasp-palm-dist-thresh", type=float, default=None)
    parser.add_argument("--pregrasp-min-finger-dist-thresh", type=float, default=None)
    parser.add_argument("--force-imitate-steps", type=int, default=None)
    parser.add_argument("--min-imitate-steps", type=int, default=None)
    parser.add_argument("--dx", nargs="+", type=float, default=[-0.02, 0.0, 0.02])
    parser.add_argument("--dy", nargs="+", type=float, default=[-0.02, 0.0, 0.02])
    parser.add_argument("--dz", nargs="+", type=float, default=[-0.02, 0.0, 0.02])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--save-best-dst", type=Path, default=None)
    parser.add_argument("--target-lift", type=float, default=0.08)
    parser.add_argument(
        "--rank-target-lift",
        action="store_true",
        help="Prefer candidates that keep contact while lifting toward --target-lift.",
    )
    parser.add_argument("--hand-dof", type=int, choices=(6, 12), default=6)
    parser.add_argument("--native-mimic-hand-qpos", action="store_true")
    parser.add_argument("--continue-after-done", action="store_true")
    parser.add_argument("--allow-pregrasp-failure", action="store_true")
    return parser.parse_args()


def _task_kwargs(
    args: argparse.Namespace,
    *,
    robot_base_yaw: float = 0.0,
    robot_base_offset: np.ndarray | None = None,
) -> dict:
    task_kwargs = {"action": "relocate", "reward_kwargs": {}}
    if robot_base_offset is None and args.robot_base_offset is not None:
        robot_base_offset = np.asarray(args.robot_base_offset, dtype=np.float32)
    if robot_base_offset is not None:
        task_kwargs["rm75_robot_base_offset"] = np.asarray(robot_base_offset, dtype=np.float32).tolist()
    if args.robot_base_rpy is not None:
        task_kwargs["rm75_robot_base_rpy"] = list(args.robot_base_rpy)
    if abs(float(robot_base_yaw)) > 1e-8:
        task_kwargs["rm75_robot_base_yaw"] = float(robot_base_yaw)
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
    if args.force_imitate_steps is not None:
        task_kwargs["rm75_force_imitate_steps"] = int(args.force_imitate_steps)
    if args.min_imitate_steps is not None:
        task_kwargs["rm75_min_imitate_steps"] = int(args.min_imitate_steps)
    return task_kwargs


def _load_source(path: Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def _copy_motion_dict(motion: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in motion.items()}


def _evaluate(
    env,
    close_val: float,
    steps: int,
    *,
    thumb_yaw: float | None = None,
    thumb_pitch: float | None = None,
    finger_close: float | None = None,
    pinky_close: float | None = None,
    mode: str = "lift_hold",
    close_steps: int = 18,
    hold_steps: int = 10,
    lift_steps: int = 35,
    lift_height: float = 0.08,
    approach_delta: np.ndarray | None = None,
    approach_steps: int = 10,
    angular_action: np.ndarray | None = None,
    angular_start_step: int = 0,
    angular_end_step: int = 1000000,
    apply_scripted_approach_pregrasp: bool = True,
    continue_after_done: bool = False,
    seed: int | None = None,
    settle_steps: int = 0,
) -> dict[str, float]:
    if mode == "approach_lift" and approach_delta is not None and apply_scripted_approach_pregrasp:
        apply_approach_pregrasp_reference(env, np.asarray(approach_delta, dtype=np.float32))
    if seed is None:
        env.reset()
    else:
        env.reset(seed=seed)
    _settle_object(env, max(int(settle_steps), 0))
    best_lift = 0.0
    final_lift = 0.0
    last_contact_lift = 0.0
    contact_steps = 0
    stable_contact_steps = 0
    max_contact_groups = 0.0
    late_contact_steps = 0
    late_stable_contact_steps = 0
    tail_contact_steps = 0
    tail_stable_contact_steps = 0
    thumb_contact_steps = 0
    non_thumb_contact_steps = 0
    max_stable_hold_steps = 0
    alive_steps = 0
    done = False
    reward_sum = 0.0
    pregrasp_checked = 0.0
    pregrasp_success = 0.0
    pregrasp_hand_jpos_err = float("inf")
    initial_object_pos = None
    pre_lift_max_obj_drift = 0.0
    final_obj_drift = 0.0
    max_xy_drift = 0.0
    final_xy_drift = 0.0
    max_tilt_err = 0.0
    max_thumb_nonthumb_angle = 0.0
    opposed_contact_steps = 0
    late_opposed_contact_steps = 0
    tail_opposed_contact_steps = 0
    best_opposed_lift = 0.0
    for _ in range(steps):
        post_step = int(env.current_step) - int(env.pregrasp_steps)
        if (
            angular_action is not None
            and post_step >= int(angular_start_step)
            and post_step <= int(angular_end_step)
        ):
            scheduled_angular = np.clip(np.asarray(angular_action, dtype=np.float32), -1.0, 1.0)
        else:
            scheduled_angular = None
        if mode == "lift_hold":
            action = _lift_hold_action(
                env,
                close_val,
                close_steps,
                hold_steps,
                lift_steps,
                lift_height,
                thumb_yaw=thumb_yaw,
                thumb_pitch=thumb_pitch,
                finger_close=finger_close,
                pinky_close=pinky_close,
                angular_action=scheduled_angular,
            )
        elif mode == "approach_lift":
            if approach_delta is None:
                approach_delta = np.array([0.0, -0.04, 0.0], dtype=np.float32)
            action = _approach_lift_action(
                env,
                close_val,
                np.asarray(approach_delta, dtype=np.float32),
                approach_steps,
                close_steps,
                hold_steps,
                lift_steps,
                lift_height,
                thumb_yaw=thumb_yaw,
                thumb_pitch=thumb_pitch,
                finger_close=finger_close,
                pinky_close=pinky_close,
                angular_action=scheduled_angular,
            )
        else:
            action = _reference_tracking_action(
                env,
                close_val,
                thumb_yaw=thumb_yaw,
                thumb_pitch=thumb_pitch,
                finger_close=finger_close,
                pinky_close=pinky_close,
                angular_action=scheduled_angular,
            )
        _, reward, done, info = env.step(action)
        if pregrasp_checked < 0.5 and int(env.current_step) >= int(env.pregrasp_steps):
            pregrasp_checked = 1.0
            pregrasp_success = float(bool(info.get("pregrasp_success", False)))
            pregrasp_hand_jpos_err = float(info.get("hand_jpos_err", float("inf")))
        object_pos = np.asarray(env.manipulated_object.get_pose().p, dtype=np.float32)
        if initial_object_pos is None:
            initial_object_pos = object_pos.copy()
        final_obj_drift = float(np.linalg.norm(object_pos - initial_object_pos))
        final_xy_drift = float(np.linalg.norm(object_pos[:2] - initial_object_pos[:2]))
        max_xy_drift = max(max_xy_drift, final_xy_drift)
        contact_count = float(info.get("contact_count", 0.0))
        final_lift = float(info.get("obj_lift", 0.0))
        best_lift = max(best_lift, final_lift)
        step_opposed_angle = 0.0
        if contact_count > 0.0:
            last_contact_lift = final_lift
            contact_details, _, _ = _contact_diagnostics(env)
            group_vectors: dict[int, list[tuple[float, np.ndarray]]] = {}
            for detail in contact_details:
                unit = detail.get("object_radial_xy_unit")
                if unit is None:
                    continue
                weight = float(detail.get("impulse_norm_sum") or detail.get("impulse_abs_sum") or 0.0)
                if weight <= 0.0:
                    continue
                group_vectors.setdefault(int(detail["group"]), []).append(
                    (weight, np.asarray(unit, dtype=np.float32))
                )
            group_dirs = {}
            for group, values in group_vectors.items():
                total_weight = sum(weight for weight, _ in values)
                mean_dir = sum(weight * vec for weight, vec in values) / max(total_weight, 1e-6)
                norm = float(np.linalg.norm(mean_dir))
                if norm > 1e-6:
                    group_dirs[group] = mean_dir / norm
            thumb_dirs = [group_dirs[group] for group in (0,) if group in group_dirs]
            nonthumb_dirs = [
                group_dirs[group]
                for group in (1, 2, 3, 4)
                if group in group_dirs
            ]
            for thumb_dir in thumb_dirs:
                for nonthumb_dir in nonthumb_dirs:
                    dot = float(np.clip(np.dot(thumb_dir, nonthumb_dir), -1.0, 1.0))
                    step_opposed_angle = max(step_opposed_angle, float(np.degrees(np.arccos(dot))))
            reward_thumb_contact = bool(info.get("thumb_contact", False))
            reward_non_thumb_contact = int(info.get("non_thumb_contact_count", 0)) > 0
            if not (reward_thumb_contact and reward_non_thumb_contact):
                step_opposed_angle = 0.0
            max_thumb_nonthumb_angle = max(max_thumb_nonthumb_angle, step_opposed_angle)
            opposed_contact = step_opposed_angle >= 120.0
            opposed_contact_steps += int(opposed_contact)
            best_opposed_lift = max(best_opposed_lift, final_lift if opposed_contact else 0.0)
        contact_steps += int(contact_count > 0.0)
        stable_contact = bool(info.get("stable_grasp_contact", False))
        stable_contact_steps += int(stable_contact)
        thumb_contact_steps += int(bool(info.get("thumb_contact", False)))
        non_thumb_contact_steps += int(int(info.get("non_thumb_contact_count", 0)) > 0)
        max_stable_hold_steps = max(max_stable_hold_steps, int(info.get("stable_contact_hold_steps", 0)))
        max_tilt_err = max(max_tilt_err, float(info.get("object_tilt_err", 0.0)))
        lift_start_step = env.pregrasp_steps + int(close_steps) + int(hold_steps)
        lifting_phase = env.current_step > lift_start_step
        tail_phase = env.current_step > lift_start_step + int(0.5 * lift_steps)
        if not lifting_phase:
            pre_lift_max_obj_drift = max(pre_lift_max_obj_drift, final_obj_drift)
        late_contact_steps += int(lifting_phase and contact_count > 0.0)
        late_stable_contact_steps += int(lifting_phase and stable_contact)
        tail_contact_steps += int(tail_phase and contact_count > 0.0)
        tail_stable_contact_steps += int(tail_phase and stable_contact)
        late_opposed_contact_steps += int(lifting_phase and contact_count > 0.0 and step_opposed_angle >= 120.0)
        tail_opposed_contact_steps += int(tail_phase and contact_count > 0.0 and step_opposed_angle >= 120.0)
        max_contact_groups = max(max_contact_groups, contact_count)
        alive_steps += 1
        reward_sum += float(reward)
        if done and not continue_after_done:
            break
    return {
        "alive": float(alive_steps),
        "pregrasp_checked": float(pregrasp_checked),
        "pregrasp_success": float(pregrasp_success),
        "pregrasp_hand_jpos_err": float(pregrasp_hand_jpos_err),
        "contact_steps": float(contact_steps),
        "stable_contact_steps": float(stable_contact_steps),
        "late_contact_steps": float(late_contact_steps),
        "late_stable_contact_steps": float(late_stable_contact_steps),
        "tail_contact_steps": float(tail_contact_steps),
        "tail_stable_contact_steps": float(tail_stable_contact_steps),
        "thumb_contact_steps": float(thumb_contact_steps),
        "non_thumb_contact_steps": float(non_thumb_contact_steps),
        "max_stable_hold_steps": float(max_stable_hold_steps),
        "max_contact_groups": float(max_contact_groups),
        "max_thumb_nonthumb_angle": float(max_thumb_nonthumb_angle),
        "opposed_contact_steps": float(opposed_contact_steps),
        "late_opposed_contact_steps": float(late_opposed_contact_steps),
        "tail_opposed_contact_steps": float(tail_opposed_contact_steps),
        "best_opposed_lift": float(best_opposed_lift),
        "best_lift": float(best_lift),
        "final_lift": float(final_lift),
        "last_contact_lift": float(last_contact_lift),
        "pre_lift_max_obj_drift": float(pre_lift_max_obj_drift),
        "final_obj_drift": float(final_obj_drift),
        "max_xy_drift": float(max_xy_drift),
        "final_xy_drift": float(final_xy_drift),
        "max_tilt_err": float(max_tilt_err),
        "reward_sum": float(reward_sum),
        "done": float(done),
    }


def _mean_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    keys = metrics_list[0].keys()
    return {key: float(np.mean([row[key] for row in metrics_list])) for key in keys}


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "hand_imitation"))

    from hand_imitation.env.rl_env.rm75_relocate_env import RM75RelocateRLEnv

    src_path = args.src or resolve_existing_motion_path(repo_root, args.seq_name, args.norm_traj, args.robot_name)
    src = _load_source(src_path)
    src["task_name"] = np.array("relocate")
    base_offset = np.asarray(args.base_offset, dtype=np.float32)
    thumb_yaws = args.thumb_yaws or args.close_vals
    thumb_pitches = args.thumb_pitches or args.close_vals
    finger_closes = args.finger_closes or args.close_vals
    pinky_closes = args.pinky_closes or [None]
    if args.approach_deltas is None:
        approach_deltas = [np.array([0.0, -0.04, 0.0], dtype=np.float32)]
    else:
        if len(args.approach_deltas) % 3 != 0:
            raise ValueError("--approach-deltas must contain 3 values per delta")
        approach_deltas = [
            np.asarray(args.approach_deltas[i : i + 3], dtype=np.float32)
            for i in range(0, len(args.approach_deltas), 3)
        ]
    if args.angular_actions is None:
        angular_actions = [np.zeros(3, dtype=np.float32)]
    else:
        if len(args.angular_actions) % 3 != 0:
            raise ValueError("--angular-actions must contain 3 values per action")
        angular_actions = [
            np.asarray(args.angular_actions[i : i + 3], dtype=np.float32)
            for i in range(0, len(args.angular_actions), 3)
        ]

    robot_base_yaws = [float(value) for value in args.robot_base_yaws]
    if args.robot_base_offsets is not None:
        if len(args.robot_base_offsets) % 3 != 0:
            raise ValueError("--robot-base-offsets must contain 3 values per offset")
        robot_base_offsets = [
            np.asarray(args.robot_base_offsets[i : i + 3], dtype=np.float32)
            for i in range(0, len(args.robot_base_offsets), 3)
        ]
    elif args.robot_base_offset is not None:
        robot_base_offsets = [np.asarray(args.robot_base_offset, dtype=np.float32)]
    else:
        robot_base_offsets = [None]

    if args.no_retarget:
        palm_offsets = [None]
    else:
        palm_offsets = [
            base_offset + np.array([dx, dy, dz], dtype=np.float32)
            for dx in args.dx
            for dy in args.dy
            for dz in args.dz
        ]

    results = []
    for palm_offset in palm_offsets:
        if palm_offset is None:
            motion = _copy_motion_dict(src)
            motion["task_name"] = np.array("relocate")
        else:
            motion = retarget_motion_dict(
                src,
                palm_offset=palm_offset,
                mode=args.retarget_mode,
                hand_dof=args.hand_dof,
                native_mimic=args.native_mimic_hand_qpos,
                seed_pinky_qpos=args.seed_pinky_qpos,
            )
            motion["task_name"] = "relocate"
        for robot_base_offset in robot_base_offsets:
            for robot_base_yaw in robot_base_yaws:
                for close_val in args.close_vals:
                    for thumb_yaw in thumb_yaws:
                        for thumb_pitch in thumb_pitches:
                            for finger_close in finger_closes:
                                for pinky_close in pinky_closes:
                                    for approach_delta in approach_deltas:
                                        for angular_action in angular_actions:
                                            repeat_metrics = []
                                            for repeat_idx in range(max(1, int(args.repeats))):
                                                candidate_motion = _copy_motion_dict(motion)
                                                if args.mode == "approach_lift" and bool(args.apply_scripted_approach_pregrasp):
                                                    candidate_motion["rm75_template_approach_delta"] = np.asarray(
                                                        approach_delta,
                                                        dtype=np.float32,
                                                    )
                                                env = RM75RelocateRLEnv(
                                                    use_gui=False,
                                                    is_eval=False,
                                                    is_vision=False,
                                                    norm_traj=args.norm_traj,
                                                    robot_name="rm75_inspire_right",
                                                    motion_file=candidate_motion,
                                                    task_kwargs=_task_kwargs(
                                                        args,
                                                        robot_base_yaw=robot_base_yaw,
                                                        robot_base_offset=robot_base_offset,
                                                    ),
                                                    no_rgb=True,
                                                    need_offscreen_render=False,
                                                )
                                                env._stage = args.stage
                                                try:
                                                    repeat_metrics.append(
                                                        _evaluate(
                                                            env,
                                                            close_val,
                                                            args.steps,
                                                            thumb_yaw=thumb_yaw,
                                                            thumb_pitch=thumb_pitch,
                                                            finger_close=finger_close,
                                                            pinky_close=pinky_close,
                                                            mode=args.mode,
                                                            close_steps=args.close_steps,
                                                            hold_steps=args.hold_steps,
                                                            lift_steps=args.lift_steps,
                                                            lift_height=args.lift_height,
                                                            approach_delta=approach_delta,
                                                            approach_steps=args.approach_steps,
                                                            angular_action=angular_action,
                                                            angular_start_step=args.angular_start_step,
                                                            angular_end_step=args.angular_end_step,
                                                            apply_scripted_approach_pregrasp=bool(
                                                                args.apply_scripted_approach_pregrasp
                                                            ),
                                                            continue_after_done=args.continue_after_done,
                                                            seed=(
                                                                None
                                                                if args.seed_start is None
                                                                else int(args.seed_start) + int(repeat_idx)
                                                            ),
                                                            settle_steps=int(args.settle_steps),
                                                        )
                                                    )
                                                finally:
                                                    env.close()
                                            metrics = _mean_metrics(repeat_metrics)
                                            pregrasp_failure = max(0.0, 1.0 - metrics["pregrasp_success"])
                                            pregrasp_err_penalty = max(
                                                0.0,
                                                metrics["pregrasp_hand_jpos_err"] - 0.075,
                                            )
                                            launch_gap = max(0.0, metrics["best_lift"] - max(metrics["last_contact_lift"], metrics["final_lift"]) - 0.003)
                                            final_gap = abs(metrics["final_lift"] - metrics["last_contact_lift"])
                                            pre_lift_drift_penalty = max(0.0, metrics["pre_lift_max_obj_drift"] - 0.025)
                                            xy_drift_penalty = max(0.0, metrics["max_xy_drift"] - 0.035)
                                            tilt_penalty = max(0.0, metrics["max_tilt_err"] - 0.25)
                                            no_tail_contact_penalty = max(0.0, 3.0 - metrics["tail_contact_steps"])
                                            no_tail_stable_penalty = max(0.0, 2.0 - metrics["tail_stable_contact_steps"])
                                            plausible_best_lift = min(metrics["best_lift"], 0.08)
                                            plausible_final_lift = min(metrics["final_lift"], 0.06)
                                            plausible_contact_lift = min(metrics["last_contact_lift"], 0.06)
                                            if metrics["max_contact_groups"] < 2.0:
                                                plausible_best_lift = 0.0
                                                plausible_final_lift = 0.0
                                                plausible_contact_lift = 0.0
                                            if metrics["stable_contact_steps"] < 2.0:
                                                plausible_final_lift *= 0.2
                                                plausible_contact_lift *= 0.5
                                            score = (
                                                metrics["alive"]
                                                + 1.5 * metrics["contact_steps"]
                                                + 7.0 * metrics["stable_contact_steps"]
                                                + 5.0 * metrics["late_contact_steps"]
                                                + 12.0 * metrics["late_stable_contact_steps"]
                                                + 12.0 * metrics["tail_contact_steps"]
                                                + 20.0 * metrics["tail_stable_contact_steps"]
                                                + 8.0 * metrics["max_thumb_nonthumb_angle"]
                                                + 30.0 * metrics["opposed_contact_steps"]
                                                + 60.0 * metrics["late_opposed_contact_steps"]
                                                + 90.0 * metrics["tail_opposed_contact_steps"]
                                                + 1800.0 * min(metrics["best_opposed_lift"], 0.05)
                                                + 4.0 * metrics["thumb_contact_steps"]
                                                + 3.0 * metrics["non_thumb_contact_steps"]
                                                + 8.0 * metrics["max_stable_hold_steps"]
                                                + 50.0 * plausible_best_lift
                                                + 2400.0 * plausible_contact_lift
                                                + 2400.0 * plausible_final_lift
                                                + 8.0 * metrics["max_contact_groups"]
                                                - 6000.0 * launch_gap
                                                - 2200.0 * final_gap
                                                - 1200.0 * pre_lift_drift_penalty
                                                - 1600.0 * xy_drift_penalty
                                                - 120.0 * tilt_penalty
                                                - 35.0 * no_tail_contact_penalty
                                                - 45.0 * no_tail_stable_penalty
                                                - 20.0 * metrics["done"]
                                                - 5000.0 * pregrasp_failure
                                                - 4000.0 * pregrasp_err_penalty
                                            )
                                            target_lift = max(float(args.target_lift), 1e-6)
                                            contact_lift_progress = min(metrics["last_contact_lift"] / target_lift, 1.0)
                                            final_lift_progress = min(metrics["final_lift"] / target_lift, 1.0)
                                            target_lift_success = float(
                                                metrics["last_contact_lift"] >= target_lift
                                                and metrics["tail_contact_steps"] >= 3.0
                                            )
                                            target_stable_lift_success = float(
                                                metrics["last_contact_lift"] >= target_lift
                                                and metrics["tail_stable_contact_steps"] >= 2.0
                                            )
                                            if args.rank_target_lift:
                                                high_lift_score = (
                                                    18000.0 * contact_lift_progress
                                                    + 7000.0 * final_lift_progress
                                                    + 35000.0 * target_lift_success
                                                    + 50000.0 * target_stable_lift_success
                                                    + 300.0 * metrics["tail_contact_steps"]
                                                    + 500.0 * metrics["tail_stable_contact_steps"]
                                                    - 15000.0 * launch_gap
                                                    - 9000.0 * xy_drift_penalty
                                                    - 3000.0 * pre_lift_drift_penalty
                                                )
                                                score += high_lift_score
                                            else:
                                                high_lift_score = 0.0
                                            if (not args.allow_pregrasp_failure) and metrics["pregrasp_success"] < 0.5:
                                                score -= 10000.0
                                            row_offset = (
                                                np.full(3, np.nan, dtype=np.float32)
                                                if palm_offset is None
                                                else palm_offset
                                            )
                                            row_robot_base_offset = (
                                                np.full(3, np.nan, dtype=np.float32)
                                                if robot_base_offset is None
                                                else robot_base_offset
                                            )
                                            row = {
                                                "score": score,
                                                "offset": row_offset,
                                                "robot_base_offset": row_robot_base_offset,
                                                "robot_base_yaw": robot_base_yaw,
                                                "approach_delta": approach_delta,
                                                "angular_action": angular_action,
                                                "close_val": close_val,
                                                "thumb_yaw": thumb_yaw,
                                                "thumb_pitch": thumb_pitch,
                                                "finger_close": finger_close,
                                                "pinky_close": np.nan if pinky_close is None else float(pinky_close),
                                                "target_lift_progress": contact_lift_progress,
                                                "target_lift_success": target_lift_success,
                                                "target_stable_lift_success": target_stable_lift_success,
                                                "high_lift_score": high_lift_score,
                                                **metrics,
                                            }
                                            results.append(row)
                                            print(
                                                "offset="
                                                f"{np.round(row_offset, 4).tolist()} base_yaw={robot_base_yaw:.4f} "
                                                f"robot_base={np.round(row_robot_base_offset, 4).tolist()} "
                                                f"approach={np.round(approach_delta, 4).tolist()} "
                                                f"angular={np.round(angular_action, 3).tolist()} close={close_val:.2f} "
                                                f"thumb=({thumb_yaw:.2f},{thumb_pitch:.2f}) fingers={finger_close:.2f} "
                                                f"pinky={row['pinky_close']:.2f} "
                                                f"pregrasp={metrics['pregrasp_success']:.1f}/{metrics['pregrasp_hand_jpos_err']:.4f} "
                                                f"alive={metrics['alive']:.1f} contact={metrics['contact_steps']:.1f} "
                                                f"stable={metrics['stable_contact_steps']:.1f} max_groups={metrics['max_contact_groups']:.1f} "
                                                f"late_contact={metrics['late_contact_steps']:.1f} late_stable={metrics['late_stable_contact_steps']:.1f} "
                                                f"tail_contact={metrics['tail_contact_steps']:.1f} tail_stable={metrics['tail_stable_contact_steps']:.1f} "
                                                f"opp_angle={metrics['max_thumb_nonthumb_angle']:.1f} opp={metrics['opposed_contact_steps']:.1f} "
                                                f"thumb={metrics['thumb_contact_steps']:.1f} nonthumb={metrics['non_thumb_contact_steps']:.1f} "
                                                f"best_lift={metrics['best_lift']:.6f} contact_lift={metrics['last_contact_lift']:.6f} "
                                                f"final_lift={metrics['final_lift']:.6f} xy={metrics['max_xy_drift']:.4f} "
                                                f"tilt={metrics['max_tilt_err']:.3f} score={score:.3f}"
                                            )

    results.sort(key=lambda row: row["score"], reverse=True)
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "score",
            "offset_x",
            "offset_y",
            "offset_z",
            "robot_base_offset_x",
            "robot_base_offset_y",
            "robot_base_offset_z",
            "robot_base_yaw",
            "approach_dx",
            "approach_dy",
            "approach_dz",
            "angular_x",
            "angular_y",
            "angular_z",
            "close_val",
            "thumb_yaw",
            "thumb_pitch",
            "finger_close",
            "pinky_close",
            "target_lift_progress",
            "target_lift_success",
            "target_stable_lift_success",
            "high_lift_score",
            "alive",
            "pregrasp_checked",
            "pregrasp_success",
            "pregrasp_hand_jpos_err",
            "contact_steps",
            "stable_contact_steps",
            "late_contact_steps",
            "late_stable_contact_steps",
            "tail_contact_steps",
            "tail_stable_contact_steps",
            "thumb_contact_steps",
            "non_thumb_contact_steps",
            "max_stable_hold_steps",
            "max_contact_groups",
            "max_thumb_nonthumb_angle",
            "opposed_contact_steps",
            "late_opposed_contact_steps",
            "tail_opposed_contact_steps",
            "best_opposed_lift",
            "best_lift",
            "final_lift",
            "last_contact_lift",
            "pre_lift_max_obj_drift",
            "final_obj_drift",
            "max_xy_drift",
            "final_xy_drift",
            "max_tilt_err",
            "reward_sum",
            "done",
        ]
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(
                    {
                        "score": row["score"],
                        "offset_x": float(row["offset"][0]),
                        "offset_y": float(row["offset"][1]),
                        "offset_z": float(row["offset"][2]),
                        "robot_base_offset_x": float(row["robot_base_offset"][0]),
                        "robot_base_offset_y": float(row["robot_base_offset"][1]),
                        "robot_base_offset_z": float(row["robot_base_offset"][2]),
                        "robot_base_yaw": float(row["robot_base_yaw"]),
                        "approach_dx": float(row["approach_delta"][0]),
                        "approach_dy": float(row["approach_delta"][1]),
                        "approach_dz": float(row["approach_delta"][2]),
                        "angular_x": float(row["angular_action"][0]),
                        "angular_y": float(row["angular_action"][1]),
                        "angular_z": float(row["angular_action"][2]),
                        "close_val": row["close_val"],
                        "thumb_yaw": row["thumb_yaw"],
                        "thumb_pitch": row["thumb_pitch"],
                        "finger_close": row["finger_close"],
                        "pinky_close": row["pinky_close"],
                        "target_lift_progress": row["target_lift_progress"],
                        "target_lift_success": row["target_lift_success"],
                        "target_stable_lift_success": row["target_stable_lift_success"],
                        "high_lift_score": row["high_lift_score"],
                        "alive": row["alive"],
                        "pregrasp_checked": row["pregrasp_checked"],
                        "pregrasp_success": row["pregrasp_success"],
                        "pregrasp_hand_jpos_err": row["pregrasp_hand_jpos_err"],
                        "contact_steps": row["contact_steps"],
                        "stable_contact_steps": row["stable_contact_steps"],
                        "late_contact_steps": row["late_contact_steps"],
                        "late_stable_contact_steps": row["late_stable_contact_steps"],
                        "tail_contact_steps": row["tail_contact_steps"],
                        "tail_stable_contact_steps": row["tail_stable_contact_steps"],
                        "thumb_contact_steps": row["thumb_contact_steps"],
                        "non_thumb_contact_steps": row["non_thumb_contact_steps"],
                        "max_stable_hold_steps": row["max_stable_hold_steps"],
                        "max_contact_groups": row["max_contact_groups"],
                        "max_thumb_nonthumb_angle": row["max_thumb_nonthumb_angle"],
                        "opposed_contact_steps": row["opposed_contact_steps"],
                        "late_opposed_contact_steps": row["late_opposed_contact_steps"],
                        "tail_opposed_contact_steps": row["tail_opposed_contact_steps"],
                        "best_opposed_lift": row["best_opposed_lift"],
                        "best_lift": row["best_lift"],
                        "final_lift": row["final_lift"],
                        "last_contact_lift": row["last_contact_lift"],
                        "pre_lift_max_obj_drift": row["pre_lift_max_obj_drift"],
                        "final_obj_drift": row["final_obj_drift"],
                        "max_xy_drift": row["max_xy_drift"],
                        "final_xy_drift": row["final_xy_drift"],
                        "max_tilt_err": row["max_tilt_err"],
                        "reward_sum": row["reward_sum"],
                        "done": row["done"],
                    }
                )
        print(f"\nSaved CSV: {args.csv_out}")

    if args.save_best_dst is not None:
        if not results:
            raise RuntimeError("No search results to save")
        best = results[0]
        if np.any(~np.isfinite(np.asarray(best["offset"], dtype=np.float32))):
            best_motion = _copy_motion_dict(src)
        else:
            best_motion = retarget_motion_dict(
                src,
                palm_offset=best["offset"],
                mode=args.retarget_mode,
                hand_dof=args.hand_dof,
                native_mimic=args.native_mimic_hand_qpos,
                seed_pinky_qpos=args.seed_pinky_qpos,
            )
        best_motion["task_name"] = np.array("relocate")
        best_motion["rm75_template_offset"] = np.asarray(best["offset"], dtype=np.float32)
        best_motion["rm75_robot_base_offset"] = np.asarray(best["robot_base_offset"], dtype=np.float32)
        best_motion["rm75_robot_base_yaw"] = np.asarray(best["robot_base_yaw"], dtype=np.float32)
        best_motion["rm75_template_approach_delta"] = np.asarray(best["approach_delta"], dtype=np.float32)
        best_motion["rm75_template_angular_action"] = np.asarray(best["angular_action"], dtype=np.float32)
        best_motion["rm75_template_angular_start_step"] = np.asarray(args.angular_start_step, dtype=np.int32)
        best_motion["rm75_template_angular_end_step"] = np.asarray(args.angular_end_step, dtype=np.int32)
        best_motion["rm75_template_close_val"] = np.asarray(best["close_val"], dtype=np.float32)
        best_motion["rm75_template_thumb_yaw"] = np.asarray(best["thumb_yaw"], dtype=np.float32)
        best_motion["rm75_template_thumb_pitch"] = np.asarray(best["thumb_pitch"], dtype=np.float32)
        best_motion["rm75_template_finger_close"] = np.asarray(best["finger_close"], dtype=np.float32)
        best_motion["rm75_template_pinky_close"] = np.asarray(best["pinky_close"], dtype=np.float32)
        best_motion["rm75_template_mode"] = np.array(args.mode)
        args.save_best_dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_best_dst, **best_motion)
        print(
            "\nSaved best retargeted trajectory: "
            f"{args.save_best_dst} offset={np.round(best['offset'], 5).tolist()} close={best['close_val']:.2f} "
            f"robot_base={np.round(best['robot_base_offset'], 5).tolist()} "
            f"base_yaw={best['robot_base_yaw']:.4f} "
            f"approach={np.round(best['approach_delta'], 5).tolist()} "
            f"angular={np.round(best['angular_action'], 5).tolist()} "
            f"thumb=({best['thumb_yaw']:.2f},{best['thumb_pitch']:.2f}) "
            f"fingers={best['finger_close']:.2f} pinky={best['pinky_close']:.2f}"
        )

    print("\nTop candidates:")
    for row in results[: args.top_k]:
        print(
            f"score={row['score']:.3f} offset={np.round(row['offset'], 5).tolist()} "
            f"robot_base={np.round(row['robot_base_offset'], 5).tolist()} "
            f"base_yaw={row['robot_base_yaw']:.4f} "
            f"approach={np.round(row['approach_delta'], 5).tolist()} "
            f"angular={np.round(row['angular_action'], 5).tolist()} "
            f"close={row['close_val']:.2f} thumb=({row['thumb_yaw']:.2f},{row['thumb_pitch']:.2f}) "
            f"fingers={row['finger_close']:.2f} pinky={row['pinky_close']:.2f} "
            f"pregrasp={row['pregrasp_success']:.1f}/{row['pregrasp_hand_jpos_err']:.4f} "
            f"alive={row['alive']:.1f} "
            f"contact={row['contact_steps']:.1f} stable={row['stable_contact_steps']:.1f} "
            f"late_contact={row['late_contact_steps']:.1f} late_stable={row['late_stable_contact_steps']:.1f} "
            f"tail_contact={row['tail_contact_steps']:.1f} tail_stable={row['tail_stable_contact_steps']:.1f} "
            f"opp_angle={row['max_thumb_nonthumb_angle']:.1f} opp={row['opposed_contact_steps']:.1f} "
            f"thumb={row['thumb_contact_steps']:.1f} nonthumb={row['non_thumb_contact_steps']:.1f} "
            f"hold={row['max_stable_hold_steps']:.0f} xy={row['max_xy_drift']:.4f} tilt={row['max_tilt_err']:.3f} "
            f"max_groups={row['max_contact_groups']:.0f} best_lift={row['best_lift']:.6f} "
            f"contact_lift={row['last_contact_lift']:.6f} final_lift={row['final_lift']:.6f}"
        )


if __name__ == "__main__":
    main()
