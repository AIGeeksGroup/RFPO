#!/usr/bin/env python3
"""Collect Allegro/UR5 reference-action demonstrations for BC warm starts."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import _init_paths
from tqdm import tqdm


DEFAULT_SEQ = "ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"


@dataclass(frozen=True)
class AllegroReferenceBCConfig:
    seq_name: str = DEFAULT_SEQ
    robot_name: str = "allegro_hand_ur5"
    norm_traj: bool = True
    stage: int = 2
    num_trajs: int = 64
    max_attempts: int = 512
    keep_failed: bool = True
    filter_success: bool = False
    success_metric: str = "sr10"
    palm_gain: float = 0.04
    hand_open_until_pregrasp: bool = True
    settle_steps: int = 0
    seed: int = 0


def _parse_args() -> argparse.Namespace:
    defaults = AllegroReferenceBCConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--seq-name", default=defaults.seq_name)
    parser.add_argument("--robot-name", default=defaults.robot_name)
    parser.add_argument("--norm-traj", action="store_true", default=defaults.norm_traj)
    parser.add_argument("--no-norm-traj", action="store_false", dest="norm_traj")
    parser.add_argument("--stage", type=int, default=defaults.stage)
    parser.add_argument("-n", "--num-trajs", type=int, default=defaults.num_trajs)
    parser.add_argument("--max-attempts", type=int, default=defaults.max_attempts)
    parser.add_argument("--keep-failed", action="store_true", default=defaults.keep_failed)
    parser.add_argument("--filter-success", action="store_true", default=defaults.filter_success)
    parser.add_argument("--success-metric", choices=("sr3", "sr10", "any"), default=defaults.success_metric)
    parser.add_argument("--palm-gain", type=float, default=defaults.palm_gain)
    parser.add_argument("--hand-open-until-pregrasp", action="store_true", default=defaults.hand_open_until_pregrasp)
    parser.add_argument("--close-during-pregrasp", action="store_false", dest="hand_open_until_pregrasp")
    parser.add_argument("--settle-steps", type=int, default=defaults.settle_steps)
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


def _hand_action_from_qpos(env, target_hand_qpos: np.ndarray) -> np.ndarray:
    qlimits = np.asarray(env.robot.get_qlimits()[env.arm_dof:], dtype=np.float32)
    target = np.asarray(target_hand_qpos, dtype=np.float32)
    if target.shape[0] != qlimits.shape[0]:
        if target.shape[0] < qlimits.shape[0]:
            pad = np.zeros(qlimits.shape[0] - target.shape[0], dtype=np.float32)
            target = np.concatenate([target, pad])
        else:
            target = target[: qlimits.shape[0]]
    denom = np.maximum(qlimits[:, 1] - qlimits[:, 0], 1e-6)
    action = 2.0 * (np.clip(target, qlimits[:, 0], qlimits[:, 1]) - qlimits[:, 0]) / denom - 1.0
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _reference_hand_qpos(env, current_step: int, *, open_until_pregrasp: bool) -> np.ndarray:
    if current_step <= env.pregrasp_steps:
        if open_until_pregrasp:
            current = np.asarray(env.robot.get_qpos()[env.arm_dof:], dtype=np.float32)
            return current * 0.0
        return np.asarray(env.pregrasp_qpos, dtype=np.float32)

    motion = getattr(env, "_motion_file", {})
    robot_qpos = np.asarray(motion.get("robot_qpos", []), dtype=np.float32)
    pregrasp_step = int(np.asarray(motion.get("pregrasp_step", 0)).item())
    if robot_qpos.ndim == 2 and robot_qpos.shape[0] > 0:
        post_idx = max(0, current_step - env.pregrasp_steps)
        qpos_idx = min(pregrasp_step + post_idx, robot_qpos.shape[0] - 1)
        return robot_qpos[qpos_idx, env.arm_dof:]
    return np.asarray(env.pregrasp_qpos, dtype=np.float32)


def _reference_action(env, args: argparse.Namespace) -> np.ndarray:
    action = np.zeros(env.action_dim, dtype=np.float32)
    if env.current_step <= env.pregrasp_steps:
        target_palm = env.cur_reference_motion["robot_pregrasp_jpos"][-1, 0]
    else:
        ref_idx = min(env.current_step - env.pregrasp_steps, len(env.cur_reference_motion["robot_jpos"]) - 1)
        target_palm = env.cur_reference_motion["robot_jpos"][ref_idx, 0]

    palm_error = np.asarray(target_palm - env.palm_link.get_pose().p, dtype=np.float32)
    action[:3] = np.clip(palm_error / max(float(args.palm_gain), 1e-6), -1.0, 1.0)
    action[3:6] = 0.0
    action[6:] = _hand_action_from_qpos(
        env,
        _reference_hand_qpos(env, env.current_step, open_until_pregrasp=bool(args.hand_open_until_pregrasp)),
    )
    return action


def _episode_success(env, metric: str) -> bool:
    sr3, sr10 = env.is_success()
    if metric == "sr3":
        return bool(sr3)
    if metric == "sr10":
        return bool(sr10)
    return bool(sr3 or sr10)


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)
    os.environ.setdefault("VIVIDEX_HEADLESS_NO_RENDER", "1")

    from hand_imitation.env.create_env import create_env

    env = create_env(
        args.seq_name,
        task_kwargs={"action": "relocate"},
        use_gui=False,
        is_eval=False,
        is_vision=False,
        norm_traj=bool(args.norm_traj),
        robot_name=args.robot_name,
    )
    env._stage = int(args.stage)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    episode_ends: list[int] = []
    episode_attempts: list[int] = []
    episode_reward: list[float] = []
    successes = 0
    accepted = 0
    attempts = 0

    try:
        with tqdm(total=args.num_trajs) as pbar:
            while accepted < args.num_trajs and attempts < args.max_attempts:
                attempts += 1
                ep_obs: list[np.ndarray] = []
                ep_actions: list[np.ndarray] = []
                ep_rewards: list[float] = []
                obs = env.reset(seed=args.seed + attempts)
                _settle_object(env, args.settle_steps)
                obs = env.get_observation()

                for _ in range(env.horizon):
                    action = _reference_action(env, args)
                    ep_obs.append(np.asarray(obs, dtype=np.float32))
                    ep_actions.append(action)
                    obs, reward, done, _info = env.step(action)
                    ep_rewards.append(float(reward))
                    if done:
                        break

                success = _episode_success(env, args.success_metric)
                keep = success or (args.keep_failed and not args.filter_success)
                if keep:
                    observations.extend(ep_obs)
                    actions.extend(ep_actions)
                    rewards.extend(ep_rewards)
                    episode_ends.append(len(observations))
                    episode_attempts.append(attempts)
                    episode_reward.append(float(np.sum(ep_rewards)))
                    accepted += 1
                    pbar.update(1)
                successes += int(success)
                pbar.set_description(f"accepted {accepted}/{attempts} success {successes}/{attempts}")
    finally:
        env.close()

    if not observations:
        raise RuntimeError("No Allegro reference episodes were collected.")

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    cfg = AllegroReferenceBCConfig(
        seq_name=args.seq_name,
        robot_name=args.robot_name,
        norm_traj=bool(args.norm_traj),
        stage=int(args.stage),
        num_trajs=int(args.num_trajs),
        max_attempts=int(args.max_attempts),
        keep_failed=bool(args.keep_failed),
        filter_success=bool(args.filter_success),
        success_metric=str(args.success_metric),
        palm_gain=float(args.palm_gain),
        hand_open_until_pregrasp=bool(args.hand_open_until_pregrasp),
        settle_steps=int(args.settle_steps),
        seed=int(args.seed),
    )
    np.savez_compressed(
        output,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
        episode_attempts=np.asarray(episode_attempts, dtype=np.int32),
        episode_reward=np.asarray(episode_reward, dtype=np.float32),
        attempts=np.asarray(attempts, dtype=np.int32),
        acceptance_rate=np.asarray(len(episode_ends) / max(attempts, 1), dtype=np.float32),
        env_name=np.asarray(args.seq_name),
        robot_name=np.asarray(args.robot_name),
        stage=np.asarray(args.stage),
        success_metric=np.asarray(args.success_metric),
        source=np.asarray("allegro_reference_scripted"),
        config_json=np.asarray(json.dumps(asdict(cfg), sort_keys=True)),
    )
    print(
        f"saved {len(episode_ends)} episodes, {len(observations)} transitions to {output}; "
        f"attempts={attempts}, success_rate={successes / max(attempts, 1):.3f}, "
        f"mean_reward={float(np.mean(episode_reward)):.3f}"
    )


if __name__ == "__main__":
    main()
