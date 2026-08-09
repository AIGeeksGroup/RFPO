"""Audit the preregistered H66 symmetric-source mechanism on Go2 states."""

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
parser.add_argument("--seed", type=int, default=20261100)
parser.add_argument("--source-seed", type=int, default=20261101)
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
from isaaclab_fpo.antithetic_inference import antithetic_action
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

    obs, _ = env.reset()
    obs = obs.to(runner.device)
    torch.manual_seed(args.source_seed)
    finite = True
    negative_sources_exact = True
    candidate_means_exact = True
    zero_dispatch_exact = True
    candidate_zero_rms = []
    positive_zero_rms = []
    source_values = 0
    for _ in range(args.rollout_steps):
        with torch.inference_mode():
            norm_obs = (
                runner.obs_normalizer(obs)
                if runner.cfg.empirical_normalization
                else obs
            )
            source = torch.randn(
                args.num_envs,
                policy.num_actions,
                device=runner.device,
                dtype=norm_obs.dtype,
            )
            negative_source = -source
            positive = policy.act_inference(norm_obs, source=source)
            negative = policy.act_inference(norm_obs, source=negative_source)
            zero = policy.act_inference(norm_obs, eval_mode="zero")
            zero_repeat = policy.act_inference(
                norm_obs, source=torch.zeros_like(source)
            )
            candidate = antithetic_action(policy, norm_obs, source)
            expected = (positive + negative) * 0.5
        finite = finite and bool(
            torch.isfinite(source).all()
            and torch.isfinite(positive).all()
            and torch.isfinite(negative).all()
            and torch.isfinite(zero).all()
            and torch.isfinite(candidate).all()
        )
        negative_sources_exact = negative_sources_exact and torch.equal(
            negative_source, -source
        )
        candidate_means_exact = candidate_means_exact and torch.equal(
            candidate, expected
        )
        zero_dispatch_exact = zero_dispatch_exact and torch.equal(zero, zero_repeat)
        candidate_zero_rms.append(normalized_rms(candidate, zero, zero))
        positive_zero_rms.append(normalized_rms(positive, zero, zero))
        source_values += source.numel()
        obs, _, _, _ = env.step(positive.to(env.device))
        obs = obs.to(runner.device)

    parameters_unchanged = all(
        torch.equal(value, actor_before[key])
        for key, value in policy.actor.state_dict().items()
    )
    mean_candidate_zero = float(np.mean(candidate_zero_rms))
    mean_positive_zero = float(np.mean(positive_zero_rms))
    gates = {
        "finite_sources_and_endpoints": finite,
        "negative_sources_bitwise_exact": negative_sources_exact,
        "candidate_float32_means_bitwise_exact": candidate_means_exact,
        "zero_dispatch_bitwise_exact": zero_dispatch_exact,
        "candidate_zero_normalized_rms_at_least_0_01": mean_candidate_zero >= 0.01,
        "candidate_closer_to_zero_than_positive": mean_candidate_zero < mean_positive_zero,
        "actor_parameters_bitwise_unchanged": parameters_unchanged,
    }
    result = {
        "hypothesis": "H66",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "task": args.task,
        "seed": args.seed,
        "source_seed": args.source_seed,
        "num_observations_per_step": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "source_value_count": source_values,
        "mean_candidate_zero_normalized_rms": mean_candidate_zero,
        "mean_positive_zero_normalized_rms": mean_positive_zero,
        "cancellation_ratio": mean_candidate_zero / mean_positive_zero,
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
