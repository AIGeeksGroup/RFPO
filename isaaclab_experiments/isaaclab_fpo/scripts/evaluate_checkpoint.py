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
parser.add_argument("--secondary-source-seed", type=int)
parser.add_argument(
    "--integration-method", choices=("euler", "midpoint"), required=True
)
parser.add_argument("--sampling-steps", type=int, required=True)
parser.add_argument(
    "--eval-modes",
    nargs="+",
    choices=(
        "zero",
        "random",
        "negative_random",
        "secondary_random",
        "iid_pair",
        "antithetic",
    ),
    default=("zero", "random"),
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
from isaaclab_fpo.antithetic_inference import antithetic_action, paired_source_action
from isaaclab_fpo.mirrored_rollouts import select_rollout_source
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
    source_hash = hashlib.sha256()
    source_value_count = 0
    source_sequence = None
    secondary_source_hash = hashlib.sha256()
    secondary_source_value_count = 0
    secondary_source_sequence = None
    actual_source_hash = hashlib.sha256()
    actual_source_value_count = 0
    actual_source_sequence = None
    stochastic_modes = (
        "random",
        "negative_random",
        "secondary_random",
        "iid_pair",
        "antithetic",
    )
    if mode in stochastic_modes:
        source_generator = torch.Generator(device=runner.device)
        source_generator.manual_seed(args.source_seed)
        source_sequence = torch.randn(
            max_steps,
            args.num_envs,
            runner.alg.policy.num_actions,
            device=runner.device,
            dtype=obs.dtype,
            generator=source_generator,
        )
        source_values = source_sequence.detach().cpu().contiguous().numpy()
        source_hash.update(str(source_values.dtype).encode())
        source_hash.update(str(source_values.shape).encode())
        source_hash.update(source_values.tobytes())
        source_value_count = source_sequence.numel()
        del source_values
        if mode in ("iid_pair", "secondary_random"):
            secondary_generator = torch.Generator(device=runner.device)
            secondary_generator.manual_seed(args.secondary_source_seed)
            secondary_source_sequence = torch.randn(
                max_steps,
                args.num_envs,
                runner.alg.policy.num_actions,
                device=runner.device,
                dtype=obs.dtype,
                generator=secondary_generator,
            )
            secondary_values = (
                secondary_source_sequence.detach().cpu().contiguous().numpy()
            )
            secondary_source_hash.update(str(secondary_values.dtype).encode())
            secondary_source_hash.update(str(secondary_values.shape).encode())
            secondary_source_hash.update(secondary_values.tobytes())
            secondary_source_value_count = secondary_source_sequence.numel()
            del secondary_values
        if mode in ("random", "negative_random", "secondary_random"):
            actual_source_sequence = select_rollout_source(
                mode, source_sequence, secondary_source_sequence
            )
            actual_values = (
                actual_source_sequence.detach().cpu().contiguous().numpy()
            )
            actual_source_hash.update(str(actual_values.dtype).encode())
            actual_source_hash.update(str(actual_values.shape).encode())
            actual_source_hash.update(actual_values.tobytes())
            actual_source_value_count = actual_source_sequence.numel()
            del actual_values

    runner.eval_mode()
    for step in range(max_steps):
        with torch.inference_mode():
            norm_obs = (
                runner.obs_normalizer(obs)
                if runner.cfg.empirical_normalization
                else obs
            )
            if mode in stochastic_modes:
                source = source_sequence[step]
                if mode in ("random", "negative_random", "secondary_random"):
                    actions = runner.alg.policy.act_inference(
                        norm_obs, source=actual_source_sequence[step]
                    )
                elif mode == "antithetic":
                    actions = antithetic_action(runner.alg.policy, norm_obs, source)
                else:
                    actions = paired_source_action(
                        runner.alg.policy,
                        norm_obs,
                        source,
                        secondary_source_sequence[step],
                    )
            else:
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
        "source_stream_sha256": (
            source_hash.hexdigest() if source_value_count else None
        ),
        "source_value_count": source_value_count,
        "actual_source_stream_sha256": (
            actual_source_hash.hexdigest() if actual_source_value_count else None
        ),
        "actual_source_value_count": actual_source_value_count,
        "source_negation_exact": (
            bool(torch.equal(actual_source_sequence, -source_sequence))
            if mode == "negative_random"
            else None
        ),
        "secondary_source_stream_sha256": (
            secondary_source_hash.hexdigest() if secondary_source_value_count else None
        ),
        "secondary_source_value_count": secondary_source_value_count,
        "endpoint_count_per_action": 2 if mode in ("iid_pair", "antithetic") else 1,
        "nfe_per_action": args.sampling_steps
        * (2 if args.integration_method == "midpoint" else 1)
        * (2 if mode in ("iid_pair", "antithetic") else 1),
    }


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.secondary_source_seed is None:
        args.secondary_source_seed = args.source_seed + 1

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
        "secondary_source_seed": args.secondary_source_seed,
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
