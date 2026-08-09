"""Audit affine antisymmetry of a frozen flow policy on fixed rollout states."""

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
parser.add_argument("--training-seed", type=int, required=True)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-v0")
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=20261650)
parser.add_argument("--source-seed", type=int, default=20261651)
parser.add_argument("--secondary-source-seed", type=int, default=20261652)
parser.add_argument("--rollout-steps", type=int, default=8)
parser.add_argument("--sampling-steps", type=int, default=32)
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


def rms(value):
    return float(torch.sqrt(torch.mean(value.double().square())))


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
    agent_cfg.policy.sampling_steps = args.sampling_steps
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
    negative_sources_exact = True
    antithetic_means_exact = True
    actor_displacements = []
    opposite_displacements = []
    iid_displacements = []
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
            opposite_source = -primary
            secondary = torch.randn(
                shape,
                device=runner.device,
                dtype=norm_obs.dtype,
                generator=secondary_generator,
            )
            positive = policy.act_inference(norm_obs, source=primary)
            opposite = policy.act_inference(norm_obs, source=opposite_source)
            independent = policy.act_inference(norm_obs, source=secondary)
            zero = policy.act_inference(norm_obs, eval_mode="zero")
            antithetic = antithetic_action(policy, norm_obs, primary)
            expected_antithetic = (positive + opposite) * 0.5

        finite = finite and all(
            bool(torch.isfinite(value).all())
            for value in (
                primary,
                secondary,
                positive,
                opposite,
                independent,
                zero,
                antithetic,
            )
        )
        negative_sources_exact = negative_sources_exact and torch.equal(
            opposite_source, -primary
        )
        antithetic_means_exact = antithetic_means_exact and torch.equal(
            antithetic, expected_antithetic
        )
        actor_displacements.append((positive - zero).detach())
        opposite_displacements.append((opposite - zero).detach())
        iid_displacements.append((independent - zero).detach())
        source_value_count += primary.numel() + secondary.numel()
        obs, _, _, _ = env.step(positive.to(env.device))
        obs = obs.to(runner.device)

    d_positive = torch.cat(actor_displacements).flatten()
    d_opposite = torch.cat(opposite_displacements).flatten()
    d_iid = torch.cat(iid_displacements).flatten()
    even = (d_positive + d_opposite) * 0.5
    odd = (d_positive - d_opposite) * 0.5
    iid_residual = (d_positive + d_iid) * 0.5
    positive_rms = rms(d_positive)
    opposite_rms = rms(d_opposite)
    even_rms = rms(even)
    odd_rms = rms(odd)
    iid_rms = rms(iid_residual)
    displacement_cosine = float(
        torch.nn.functional.cosine_similarity(
            d_positive.double(), d_opposite.double(), dim=0
        )
    )
    antithetic_ratio = even_rms / positive_rms
    iid_ratio = iid_rms / positive_rms
    actor_unchanged = all(
        torch.equal(value, actor_before[key])
        for key, value in policy.actor.state_dict().items()
    )
    gates = {
        "finite_sources_and_endpoints": finite,
        "negative_sources_bitwise_exact": negative_sources_exact,
        "antithetic_float32_means_bitwise_exact": antithetic_means_exact,
        "actor_parameters_bitwise_unchanged": actor_unchanged,
        "displacement_cosine_at_most_negative_0_90": displacement_cosine <= -0.90,
        "antithetic_residual_ratio_at_most_0_20": antithetic_ratio <= 0.20,
        "antithetic_ratio_at_most_0_30_iid": antithetic_ratio <= 0.30 * iid_ratio,
    }
    result = {
        "hypothesis": "H75",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "training_seed": args.training_seed,
        "task": args.task,
        "seed": args.seed,
        "source_seed": args.source_seed,
        "secondary_source_seed": args.secondary_source_seed,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "sampling_steps": args.sampling_steps,
        "source_value_count": source_value_count,
        "action_value_count": d_positive.numel(),
        "displacement_cosine": displacement_cosine,
        "positive_displacement_rms": positive_rms,
        "opposite_displacement_rms": opposite_rms,
        "even_displacement_rms": even_rms,
        "odd_displacement_rms": odd_rms,
        "iid_pair_residual_rms": iid_rms,
        "antithetic_residual_ratio": antithetic_ratio,
        "iid_residual_ratio": iid_ratio,
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
