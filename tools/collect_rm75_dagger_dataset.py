#!/usr/bin/env python3
"""Collect RM75 DAgger states by executing a policy and labeling scripted expert actions."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import _init_paths  # noqa: F401

from algos.rl.fpo_core import FPOPolicyConfig, FPOStatePolicy
from tools.scripted_rm75_grasp_smoke import _approach_lift_action, apply_approach_pregrasp_reference


DEFAULT_SEQ = "ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"


@dataclass(frozen=True)
class DAggerConfig:
    seq_name: str
    checkpoint: str
    num_trajs: int
    max_steps: int
    execute_expert_blend: float
    settle_steps: int
    reset_settle_steps: int
    force_imitate_steps: int
    seed: int
    object_scale: float
    robot_base_offset: tuple[float, float, float]
    robot_base_rpy: tuple[float, float, float]
    robot_base_yaw: float
    approach_delta: tuple[float, float, float]
    angular_action: tuple[float, float, float]
    angular_start_step: int
    angular_end_step: int
    close_steps: int
    hold_steps: int
    lift_steps: int
    lift_height: float
    close_val: float
    thumb_yaw: float
    thumb_pitch: float
    finger_close: float
    pinky_close: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-name", default=DEFAULT_SEQ)
    parser.add_argument("--num-trajs", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=205)
    parser.add_argument("--execute-expert-blend", type=float, default=0.0)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--reset-settle-steps", type=int, default=8)
    parser.add_argument("--force-imitate-steps", type=int, default=205)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--robot-base-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--robot-base-rpy", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--robot-base-yaw", type=float, default=0.0)
    parser.add_argument("--no-done-on-pregrasp-failure", action="store_true")
    parser.add_argument("--no-done-on-norm-success", action="store_true")
    parser.add_argument("--ignore-pinky-contact", action="store_true")
    parser.add_argument("--disable-pinky-action", action="store_true")
    parser.add_argument("--pinky-action-value", type=float, default=0.0)
    parser.add_argument("--success-min-non-thumb-contacts", type=int, default=1)
    parser.add_argument("--required-non-thumb-contacts", type=int, default=1)
    parser.add_argument("--obj-com-done-thresh", type=float, default=0.35)
    parser.add_argument("--no-contact-grace-steps", type=int, default=205)
    parser.add_argument("--bad-push-done", action="store_true")
    parser.add_argument("--approach-delta", nargs=3, type=float, default=[0.01, -0.06, -0.02])
    parser.add_argument("--angular-action", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--angular-start-step", type=int, default=0)
    parser.add_argument("--angular-end-step", type=int, default=0)
    parser.add_argument("--approach-steps", type=int, default=14)
    parser.add_argument("--close-steps", type=int, default=30)
    parser.add_argument("--hold-steps", type=int, default=36)
    parser.add_argument("--lift-steps", type=int, default=140)
    parser.add_argument("--lift-height", type=float, default=0.14)
    parser.add_argument("--close-val", type=float, default=0.30)
    parser.add_argument("--thumb-yaw", type=float, default=0.90)
    parser.add_argument("--thumb-pitch", type=float, default=0.85)
    parser.add_argument("--finger-close", type=float, default=0.75)
    parser.add_argument("--pinky-close", type=float, default=None)
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
        "rm75_norm_success_lift_thresh": 0.08,
        "rm75_norm_success_requires_contact": True,
        "rm75_success_min_non_thumb_contacts": int(args.success_min_non_thumb_contacts),
        "rm75_ignore_pinky_contact": bool(args.ignore_pinky_contact),
        "rm75_disable_pinky_action": bool(args.disable_pinky_action),
        "rm75_pinky_action_value": float(args.pinky_action_value),
        "reward_kwargs": {
            "obj_com_done_thresh": float(args.obj_com_done_thresh),
            "no_contact_grace_steps": int(args.no_contact_grace_steps),
            "bad_push_done": bool(args.bad_push_done),
            "bad_push_min_step": 6,
            "bad_push_drift_thresh": 0.08,
            "bad_push_tilt_thresh": 0.75,
            "required_non_thumb_contacts": int(args.required_non_thumb_contacts),
        },
    }


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


def _expert_action(env, args: argparse.Namespace) -> np.ndarray:
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


@torch.no_grad()
def _policy_action(policy: FPOStatePolicy, obs: np.ndarray, device: torch.device) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
    return policy.act(obs_tensor, deterministic=True).detach().cpu().numpy()[0]


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("VIVIDEX_HEADLESS_NO_RENDER", "1")
    os.environ.setdefault("VIVIDEX_RM75_NATIVE_HAND_CONTROL", "1")
    os.environ.setdefault("VIVIDEX_RM75_HAND_DOF", "12")

    from hand_imitation.env.create_rm75_env import create_rm75_env

    device = _device(args.device)
    policy = _load_policy(args.checkpoint, device)
    env = create_rm75_env(
        args.seq_name,
        use_gui=False,
        is_eval=False,
        is_vision=False,
        norm_traj=True,
        robot_name="rm75_inspire_right",
        task_kwargs=_task_kwargs(args),
    )
    env._stage = 0
    apply_approach_pregrasp_reference(env, np.asarray(args.approach_delta, dtype=np.float32))

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    episode_ends: list[int] = []
    episode_contact_steps: list[int] = []
    episode_stable_steps: list[int] = []
    episode_best_lift: list[float] = []
    blend = float(np.clip(args.execute_expert_blend, 0.0, 1.0))

    try:
        for episode in tqdm(range(int(args.num_trajs))):
            obs = env.reset(seed=int(args.seed) + episode)
            _settle_object(env, int(args.settle_steps))
            obs = env.get_observation()
            contact_steps = 0
            stable_steps = 0
            best_lift = 0.0
            for _ in range(int(args.max_steps)):
                expert_action = _expert_action(env, args)
                policy_action = _policy_action(policy, obs, device)
                execute_action = np.clip((1.0 - blend) * policy_action + blend * expert_action, -1.0, 1.0)
                observations.append(np.asarray(obs, dtype=np.float32))
                actions.append(np.asarray(expert_action, dtype=np.float32))
                obs, reward, done, info = env.step(execute_action)
                rewards.append(float(reward))
                contact_steps += int(float(info.get("contact_count", 0.0)) > 0.0)
                stable_steps += int(bool(info.get("stable_grasp_contact", False)))
                best_lift = max(best_lift, float(info.get("obj_lift", 0.0)))
                if done:
                    break
            episode_ends.append(len(observations))
            episode_contact_steps.append(contact_steps)
            episode_stable_steps.append(stable_steps)
            episode_best_lift.append(best_lift)
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = DAggerConfig(
        seq_name=args.seq_name,
        checkpoint=str(args.checkpoint),
        num_trajs=int(args.num_trajs),
        max_steps=int(args.max_steps),
        execute_expert_blend=blend,
        settle_steps=int(args.settle_steps),
        reset_settle_steps=int(args.reset_settle_steps),
        force_imitate_steps=int(args.force_imitate_steps),
        seed=int(args.seed),
        object_scale=float(args.object_scale),
        robot_base_offset=tuple(float(v) for v in args.robot_base_offset),
        robot_base_rpy=tuple(float(v) for v in args.robot_base_rpy),
        robot_base_yaw=float(args.robot_base_yaw),
        approach_delta=tuple(float(v) for v in args.approach_delta),
        angular_action=tuple(float(v) for v in args.angular_action),
        angular_start_step=int(args.angular_start_step),
        angular_end_step=int(args.angular_end_step),
        close_steps=int(args.close_steps),
        hold_steps=int(args.hold_steps),
        lift_steps=int(args.lift_steps),
        lift_height=float(args.lift_height),
        close_val=float(args.close_val),
        thumb_yaw=float(args.thumb_yaw),
        thumb_pitch=float(args.thumb_pitch),
        finger_close=float(args.finger_close),
        pinky_close=None if args.pinky_close is None else float(args.pinky_close),
    )
    np.savez_compressed(
        args.output,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
        episode_contact_steps=np.asarray(episode_contact_steps, dtype=np.int32),
        episode_stable_steps=np.asarray(episode_stable_steps, dtype=np.int32),
        episode_best_lift=np.asarray(episode_best_lift, dtype=np.float32),
        source=np.asarray("rm75_dagger_policy_states"),
        config_json=np.asarray(str(asdict(cfg))),
    )
    print(
        f"saved {len(episode_ends)} episodes, {len(observations)} transitions to {args.output}; "
        f"mean_contact_steps={np.mean(episode_contact_steps):.2f}, "
        f"mean_stable_steps={np.mean(episode_stable_steps):.2f}, "
        f"mean_best_lift={np.mean(episode_best_lift):.4f}"
    )


if __name__ == "__main__":
    main()
