#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

import numpy as np
import yaml
import _init_paths
from tqdm import tqdm


def _resolve_checkpoint_inputs(args) -> tuple[str, str]:
    if args.checkpoint_file:
        checkpoint_path = args.checkpoint_file
        config_root = args.config_dir or args.checkpoint_dir
        if not config_root:
            raise ValueError("--config_dir or --checkpoint_dir is required with --checkpoint_file")
    else:
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir is required when --checkpoint_file is not provided")
        checkpoint_path = os.path.join(args.checkpoint_dir, "restore_checkpoint.zip")
        config_root = args.config_dir or args.checkpoint_dir
    return checkpoint_path, os.path.join(config_root, "exp_config.yaml")


def _predict_ppo(policy, obs):
    obs_batch = obs[None, :].astype(np.float32)
    try:
        action = policy.predict(observation=obs_batch, deterministic=True)[0]
    except Exception:
        fix_obs = np.zeros((1, 396), dtype=np.float32)
        fix_obs[:, :367] = obs_batch[:, :367]
        fix_obs[:, 367:370] = obs_batch[:, 364:367]
        fix_obs[:, 370:] = obs_batch[:, 367:]
        action = policy.predict(observation=fix_obs, deterministic=True)[0]
    if len(action.shape) > 1:
        action = action[0]
    return np.asarray(action, dtype=np.float32)


def _episode_success(env, metric: str) -> bool:
    sr3, sr10 = env.is_success()
    if metric == "sr3":
        return bool(sr3)
    if metric == "sr10":
        return bool(sr10)
    if metric == "any":
        return bool(sr3 or sr10)
    raise ValueError(f"Unknown success metric: {metric}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--checkpoint_dir")
    parser.add_argument("--checkpoint_file")
    parser.add_argument("--config_dir")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-n", "--num_trajs", type=int, default=100)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--success_metric", choices=["sr3", "sr10", "any"], default="sr10")
    parser.add_argument("--keep_failed", action="store_true")
    parser.add_argument("--norm_traj", action="store_true")
    parser.add_argument("--real_robot", action="store_true")
    args = parser.parse_args()

    checkpoint_path, config_path = _resolve_checkpoint_inputs(args)
    config = yaml.safe_load(open(config_path, "r"))
    from stable_baselines3 import PPO

    policy = PPO.load(checkpoint_path)

    env_name = config["params"]["env"]["name"]
    env_cfg = config["params"]["env"]
    task_kwargs = env_cfg["task_kwargs"]
    from hand_imitation.env.create_env import create_env

    env = create_env(
        name=env_name,
        task_kwargs=task_kwargs,
        use_gui=False,
        is_eval=True,
        is_vision=False,
        is_demo_rollout=False,
        is_real_robot=args.real_robot,
        norm_traj=args.norm_traj,
        robot_name=env_cfg.get("robot_name", "allegro_hand_ur5"),
    )
    env._stage = int(args.stage)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    episode_ends: list[int] = []
    successes = 0
    accepted = 0
    attempts = 0

    with tqdm(total=args.num_trajs) as pbar:
        while accepted < args.num_trajs:
            attempts += 1
            ep_obs: list[np.ndarray] = []
            ep_actions: list[np.ndarray] = []
            ep_rewards: list[float] = []
            obs = env.reset()
            for _ in range(env.horizon):
                action = _predict_ppo(policy, obs)
                ep_obs.append(np.asarray(obs, dtype=np.float32))
                ep_actions.append(action)
                obs, reward, done, _info = env.step(action)
                ep_rewards.append(float(reward))
                if done:
                    break

            success = _episode_success(env, args.success_metric)
            if success or args.keep_failed:
                observations.extend(ep_obs)
                actions.extend(ep_actions)
                rewards.extend(ep_rewards)
                episode_ends.append(len(observations))
                accepted += 1
                pbar.update(1)
            if success:
                successes += 1
            pbar.set_description(f"accepted {accepted}, success {successes}/{attempts}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
        env_name=np.asarray(env_name),
        stage=np.asarray(args.stage),
        success_metric=np.asarray(args.success_metric),
        teacher_checkpoint=np.asarray(checkpoint_path),
        teacher_config=np.asarray(config_path),
    )
    print(f"saved {len(episode_ends)} episodes, {len(observations)} transitions to {args.output}")


if __name__ == "__main__":
    main()
