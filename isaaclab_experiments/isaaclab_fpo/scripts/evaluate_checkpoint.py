"""Evaluate one locomotion checkpoint with exact per-environment episode accounting."""

# ruff: noqa: E402

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-v0")
parser.add_argument("--num-envs", type=int, default=50)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=20261061)
parser.add_argument("--source-seed", type=int, default=20261060)
parser.add_argument(
    "--integration-method", choices=("euler", "midpoint"), required=True
)
parser.add_argument("--sampling-steps", type=int, required=True)
parser.add_argument(
    "--eval-modes", nargs="+", choices=("zero", "random"), default=("zero", "random")
)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.episodes != args.num_envs:
    parser.error(
        "H60 requires exactly one episode per environment: --episodes must equal --num-envs"
    )

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab_fpo.patches import apply_isaaclab_patches

apply_isaaclab_patches()

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import whole_body_tracking  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_fpo import FpoRslRlVecEnvWrapper
from isaaclab_fpo.runners import OnPolicyRunner
from isaaclab_fpo.task_cfgs import TASK_CONFIGS


def evaluate_mode(runner, env, mode):
    torch.manual_seed(args.source_seed)
    np.random.seed(args.source_seed)
    random.seed(args.source_seed)

    obs, _ = env.reset()
    obs = obs.to(runner.device)
    initial_obs = obs.detach().cpu().contiguous().numpy()
    initial_obs_hash = hashlib.sha256()
    initial_obs_hash.update(str(initial_obs.dtype).encode())
    initial_obs_hash.update(str(initial_obs.shape).encode())
    initial_obs_hash.update(initial_obs.tobytes())
    rewards = torch.zeros(args.num_envs, device=runner.device)
    lengths = torch.zeros(args.num_envs, dtype=torch.long, device=runner.device)
    completed = torch.zeros(args.num_envs, dtype=torch.bool, device=runner.device)
    episode_returns = torch.full((args.num_envs,), torch.nan, device=runner.device)
    episode_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=runner.device)
    max_steps = int(env.max_episode_length) * 2
    actions_finite = True

    runner.eval_mode()
    for _ in range(max_steps):
        with torch.inference_mode():
            norm_obs = (
                runner.obs_normalizer(obs)
                if runner.cfg.empirical_normalization
                else obs
            )
            actions = runner.alg.policy.act_inference(norm_obs, eval_mode=mode)
        actions_finite = actions_finite and bool(torch.isfinite(actions).all())
        obs, step_rewards, dones, _ = env.step(actions.to(env.device))
        obs = obs.to(runner.device)
        active = ~completed
        rewards[active] += step_rewards.to(runner.device)[active]
        lengths[active] += 1
        newly_done = active & (dones.to(runner.device) > 0)
        episode_returns[newly_done] = rewards[newly_done]
        episode_lengths[newly_done] = lengths[newly_done]
        completed |= newly_done
        if bool(completed.all()):
            break

    if not bool(completed.all()):
        missing = (~completed).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(f"environments did not finish: {missing}")

    values = episode_returns.cpu().numpy()
    lens = episode_lengths.cpu().numpy()
    return {
        "episode_returns": values.tolist(),
        "episode_lengths": lens.tolist(),
        "mean_reward": float(values.mean()),
        "std_reward": float(values.std(ddof=1)),
        "sem_reward": float(values.std(ddof=1) / math.sqrt(len(values))),
        "mean_length": float(lens.mean()),
        "actions_finite": actions_finite,
        "episodes": len(values),
        "initial_observation_sha256": initial_obs_hash.hexdigest(),
    }


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = TASK_CONFIGS[args.task]()
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.policy.integration_method = args.integration_method
    agent_cfg.policy.sampling_steps = args.sampling_steps

    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint, load_optimizer=False)
    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "task": args.task,
        "seed": args.seed,
        "source_seed": args.source_seed,
        "num_envs": args.num_envs,
        "episodes_per_mode": args.episodes,
        "integration_method": args.integration_method,
        "sampling_steps": args.sampling_steps,
        "nfe": args.sampling_steps
        * (2 if args.integration_method == "midpoint" else 1),
        "modes": {},
    }
    for mode in args.eval_modes:
        results["modes"][mode] = evaluate_mode(runner, env, mode)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
