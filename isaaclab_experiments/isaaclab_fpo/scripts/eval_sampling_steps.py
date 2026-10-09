# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate checkpoints across flow sampling step counts."""

import argparse
import sys

from isaaclab.app import AppLauncher

from isaaclab_fpo import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Sweep flow sampling steps at eval time.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument(
    "--model",
    type=str,
    action="append",
    required=True,
    help="Model checkpoint as label=path (repeatable).",
)
parser.add_argument(
    "--sampling_steps",
    type=int,
    nargs="+",
    default=[64, 32, 16, 8, 4, 1],
)
parser.add_argument("--eval_episodes", type=int, default=10)
parser.add_argument(
    "--eval_modes",
    type=str,
    nargs="+",
    default=["zero", "random"],
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_fpo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_fpo.patches import apply_isaaclab_patches

apply_isaaclab_patches()

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_fpo import FpoRslRlOnPolicyRunnerCfg, FpoRslRlVecEnvWrapper
from isaaclab_fpo.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def parse_checkpoint_arg(spec: str) -> tuple[str, str]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), path.strip()
    import os

    base = os.path.basename(spec)
    return base.replace(".pt", ""), spec


def evaluate_with_steps(runner: OnPolicyRunner, sampling_steps: int, eval_mode: str) -> dict:
    policy = runner.alg.policy
    old_steps = policy.sampling_steps
    policy.sampling_steps = sampling_steps

    obs, _ = runner.env.reset()
    obs = obs.to(runner.device)

    num_episodes = args_cli.eval_episodes
    mode_rewards = []
    episode_reward = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)
    episode_length = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)
    max_episode_length = getattr(runner.env, "max_episode_length", 1000)
    episodes_per_env = torch.zeros(runner.env.num_envs, dtype=torch.long, device=runner.device)
    target_episodes_per_env = max(1, num_episodes // runner.env.num_envs)
    eval_fixed_seed = getattr(runner.cfg, "flow_eval_fixed_seed", 12345)

    while (episodes_per_env < target_episodes_per_env).any():
        with torch.no_grad():
            norm_obs = runner.obs_normalizer(obs) if runner.cfg.empirical_normalization else obs
            actions = policy.act_inference(
                norm_obs,
                eval_mode=eval_mode,
                eval_fixed_seed=eval_fixed_seed,
            )
        obs, rewards, dones, _ = runner.env.step(actions.to(runner.env.device))
        obs = obs.to(runner.device)
        episode_reward += rewards.to(runner.device)
        episode_length += 1
        done_mask = (dones > 0) | (episode_length >= max_episode_length)
        if done_mask.any():
            for idx in done_mask.nonzero(as_tuple=False).squeeze(-1):
                if episodes_per_env[idx] < target_episodes_per_env:
                    mode_rewards.append(episode_reward[idx].item())
                    episodes_per_env[idx] += 1
            episode_reward[done_mask] = 0
            episode_length[done_mask] = 0

    policy.sampling_steps = old_steps
    return {
        "mean_reward": float(np.mean(mode_rewards)) if mode_rewards else 0.0,
        "std_reward": float(np.std(mode_rewards)) if mode_rewards else 0.0,
        "num_episodes": len(mode_rewards),
    }


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: FpoRslRlOnPolicyRunnerCfg = cli_args.parse_fpo_cfg(task_name, args_cli)
    agent_cfg.eval_episodes = args_cli.eval_episodes

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.eval_mode()

    checkpoints = [parse_checkpoint_arg(spec) for spec in args_cli.model]

    print("\n" + "=" * 90)
    print("Flow sampling-steps sweep (PostEval-style rollouts)")
    print(f"  task={args_cli.task}  envs={args_cli.num_envs}  episodes={args_cli.eval_episodes}")
    print(f"  steps={args_cli.sampling_steps}  modes={args_cli.eval_modes}")
    print("=" * 90)

    all_results = {}
    for label, ckpt_path in checkpoints:
        print(f"\n>>> Loading {label}: {ckpt_path}")
        runner.load(ckpt_path)
        all_results[label] = {}
        for steps in args_cli.sampling_steps:
            all_results[label][steps] = {}
            for mode in args_cli.eval_modes:
                res = evaluate_with_steps(runner, steps, mode)
                all_results[label][steps][mode] = res
                print(
                    f"  steps={steps:3d}  {mode:6s}  "
                    f"reward={res['mean_reward']:7.2f} ± {res['std_reward']:5.2f}  "
                    f"(n={res['num_episodes']})"
                )

    print("\n" + "=" * 90)
    print("SUMMARY TABLE (mean reward)")
    header = f"{'model':<14}" + "".join(f"{s:>8}" for s in args_cli.sampling_steps)
    print(header)
    for label in all_results:
        for mode in args_cli.eval_modes:
            row = f"{label + '/' + mode:<14}"
            for steps in args_cli.sampling_steps:
                row += f"{all_results[label][steps][mode]['mean_reward']:8.2f}"
            print(row)

    print("\nDROP FROM 64 STEPS (same model/mode)")
    for label in all_results:
        for mode in args_cli.eval_modes:
            base = all_results[label][64][mode]["mean_reward"] if 64 in all_results[label] else None
            if base is None:
                continue
            drops = []
            for steps in args_cli.sampling_steps:
                r = all_results[label][steps][mode]["mean_reward"]
                drops.append(f"{steps}:{r - base:+.2f}")
            print(f"  {label}/{mode}: " + "  ".join(drops))
    print("=" * 90 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
