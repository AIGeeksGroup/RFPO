"""Audit antithetic versus IID endpoint averaging on fixed Go2 states."""

# ruff: noqa: E402

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-v0")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=20261110)
parser.add_argument("--source-seed", type=int, default=20261111)
parser.add_argument("--secondary-source-seed", type=int, default=20261112)
parser.add_argument("--rollout-steps", type=int, default=8)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.output.exists():
    parser.error(f"refusing to overwrite {args.output}")

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
from isaaclab_fpo.runners import OnPolicyRunner
from isaaclab_fpo.task_cfgs import TASK_CONFIGS


def normalized_rms(left, right, reference):
    numerator = torch.sqrt(torch.mean((left - right).double().square()))
    denominator = torch.sqrt(torch.mean(reference.double().square())).clamp_min(1e-12)
    return float(numerator / denominator)


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = TASK_CONFIGS[args.task]()
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.policy.integration_method = "euler"
    agent_cfg.policy.sampling_steps = 64
    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint, load_optimizer=False)
    runner.eval_mode()
    policy = runner.alg.policy
    actor_before = copy.deepcopy(policy.actor.state_dict())

    primary_generator = torch.Generator(device=runner.device)
    primary_generator.manual_seed(args.source_seed)
    secondary_generator = torch.Generator(device=runner.device)
    secondary_generator.manual_seed(args.secondary_source_seed)
    obs, _ = env.reset()
    obs = obs.to(runner.device)
    finite = True
    zero_dispatch_exact = True
    antithetic_means_exact = True
    iid_means_exact = True
    primary_secondary_differ = False
    positive_zero_rms = []
    iid_zero_rms = []
    antithetic_zero_rms = []
    source_value_count = 0

    for _ in range(args.rollout_steps):
        with torch.inference_mode():
            norm_obs = (
                runner.obs_normalizer(obs)
                if runner.cfg.empirical_normalization
                else obs
            )
            shape = (args.num_envs, policy.num_actions)
            primary = torch.randn(
                shape,
                device=runner.device,
                dtype=norm_obs.dtype,
                generator=primary_generator,
            )
            secondary = torch.randn(
                shape,
                device=runner.device,
                dtype=norm_obs.dtype,
                generator=secondary_generator,
            )
            positive = policy.act_inference(norm_obs, source=primary)
            negative = policy.act_inference(norm_obs, source=-primary)
            independent = policy.act_inference(norm_obs, source=secondary)
            zero = policy.act_inference(norm_obs, eval_mode="zero")
            zero_repeat = policy.act_inference(
                norm_obs, source=torch.zeros_like(primary)
            )
            antithetic = antithetic_action(policy, norm_obs, primary)
            iid = paired_source_action(policy, norm_obs, primary, secondary)
            expected_antithetic = (positive + negative) * 0.5
            expected_iid = (positive + independent) * 0.5

        finite = finite and all(
            bool(torch.isfinite(value).all())
            for value in (
                primary,
                secondary,
                positive,
                negative,
                independent,
                zero,
                antithetic,
                iid,
            )
        )
        zero_dispatch_exact = zero_dispatch_exact and torch.equal(zero, zero_repeat)
        antithetic_means_exact = antithetic_means_exact and torch.equal(
            antithetic, expected_antithetic
        )
        iid_means_exact = iid_means_exact and torch.equal(iid, expected_iid)
        primary_secondary_differ = primary_secondary_differ or not torch.equal(
            primary, secondary
        )
        positive_zero_rms.append(normalized_rms(positive, zero, zero))
        iid_zero_rms.append(normalized_rms(iid, zero, zero))
        antithetic_zero_rms.append(normalized_rms(antithetic, zero, zero))
        source_value_count += primary.numel() + secondary.numel()
        obs, _, _, _ = env.step(positive.to(env.device))
        obs = obs.to(runner.device)

    actor_unchanged = all(
        torch.equal(value, actor_before[key])
        for key, value in policy.actor.state_dict().items()
    )
    mean_positive = float(np.mean(positive_zero_rms))
    mean_iid = float(np.mean(iid_zero_rms))
    mean_antithetic = float(np.mean(antithetic_zero_rms))
    gates = {
        "finite_sources_and_endpoints": finite,
        "primary_and_secondary_sources_differ": primary_secondary_differ,
        "zero_dispatch_bitwise_exact": zero_dispatch_exact,
        "antithetic_float32_means_bitwise_exact": antithetic_means_exact,
        "iid_float32_means_bitwise_exact": iid_means_exact,
        "antithetic_closer_to_zero_than_iid": mean_antithetic < mean_iid,
        "actor_parameters_bitwise_unchanged": actor_unchanged,
    }
    result = {
        "hypothesis": "H67",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "task": args.task,
        "seed": args.seed,
        "source_seed": args.source_seed,
        "secondary_source_seed": args.secondary_source_seed,
        "num_observations_per_step": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "source_value_count": source_value_count,
        "mean_positive_zero_normalized_rms": mean_positive,
        "mean_iid_zero_normalized_rms": mean_iid,
        "mean_antithetic_zero_normalized_rms": mean_antithetic,
        "iid_cancellation_ratio": mean_iid / mean_positive,
        "antithetic_cancellation_ratio": mean_antithetic / mean_positive,
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
