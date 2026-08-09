"""Audit equal-NFE flow solvers on fixed observations and sources."""

# ruff: noqa: E402

import argparse
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
parser.add_argument("--seed", type=int, default=20261060)
parser.add_argument("--rollout-steps", type=int, default=8)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

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


def integrate(policy, observations, source, method, steps):
    times = torch.linspace(1.0, 0.0, steps + 1, device=observations.device)
    calls = 0

    def count_call(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = policy.actor.register_forward_hook(count_call)
    try:
        if method == "euler":
            endpoint = policy._integrate_flow(
                observations, source.clone(), times[:-1], times[1:] - times[:-1], steps
            )
        else:
            endpoint = policy._integrate_flow_midpoint(
                observations, source.clone(), times[:-1], times[1:] - times[:-1], steps
            )
    finally:
        handle.remove()
    return policy.actor_scale * endpoint, calls


def mse(left, right):
    return float(torch.mean((left - right) ** 2))


def diversity(actions):
    return {
        "mean_element_std": float(actions.std(dim=0).mean()),
        "mean_pairwise_distance": float(torch.pdist(actions).mean()),
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
    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint, load_optimizer=False)
    runner.eval_mode()

    obs, _ = env.reset()
    obs = obs.to(runner.device)
    for _ in range(args.rollout_steps):
        with torch.inference_mode():
            norm_obs = (
                runner.obs_normalizer(obs)
                if runner.cfg.empirical_normalization
                else obs
            )
            actions = runner.alg.policy.act_inference(
                norm_obs,
                eval_mode="zero",
                integration_method="euler",
                sampling_steps=64,
            )
        obs, _, _, _ = env.step(actions.to(env.device))
        obs = obs.to(runner.device)

    with torch.inference_mode():
        observations = (
            runner.obs_normalizer(obs) if runner.cfg.empirical_normalization else obs
        )
        generator = torch.Generator(device=runner.device).manual_seed(args.seed)
        sources = {
            "zero": torch.zeros(
                args.num_envs, runner.alg.policy.num_actions, device=runner.device
            ),
            "random": torch.randn(
                args.num_envs,
                runner.alg.policy.num_actions,
                device=runner.device,
                generator=generator,
            ),
        }
        results = {}
        for mode, source in sources.items():
            control, control_nfe = integrate(
                runner.alg.policy, observations, source, "euler", 64
            )
            candidate, candidate_nfe = integrate(
                runner.alg.policy, observations, source, "midpoint", 32
            )
            reference, reference_nfe = integrate(
                runner.alg.policy, observations, source, "euler", 256
            )
            mode_result = {
                "control_mse": mse(control, reference),
                "candidate_mse": mse(candidate, reference),
                "mse_ratio": mse(candidate, reference) / mse(control, reference),
                "control_nfe": control_nfe,
                "candidate_nfe": candidate_nfe,
                "reference_nfe": reference_nfe,
                "finite": bool(
                    torch.isfinite(control).all()
                    and torch.isfinite(candidate).all()
                    and torch.isfinite(reference).all()
                ),
            }
            if mode == "random":
                repeated_obs = observations[:1].expand(args.num_envs, -1)
                control_diverse, _ = integrate(
                    runner.alg.policy, repeated_obs, source, "euler", 64
                )
                candidate_diverse, _ = integrate(
                    runner.alg.policy, repeated_obs, source, "midpoint", 32
                )
                control_diversity = diversity(control_diverse)
                candidate_diversity = diversity(candidate_diverse)
                mode_result["control_diversity"] = control_diversity
                mode_result["candidate_diversity"] = candidate_diversity
                mode_result["diversity_ratios"] = {
                    key: candidate_diversity[key] / control_diversity[key]
                    for key in control_diversity
                }
            results[mode] = mode_result

    gates = {
        "zero_finite": results["zero"]["finite"],
        "random_finite": results["random"]["finite"],
        "zero_mse_ratio_at_most_0_25": results["zero"]["mse_ratio"] <= 0.25,
        "random_mse_ratio_at_most_0_25": results["random"]["mse_ratio"] <= 0.25,
        "control_nfe_exact": all(
            result["control_nfe"] == 64 for result in results.values()
        ),
        "candidate_nfe_exact": all(
            result["candidate_nfe"] == 64 for result in results.values()
        ),
        "reference_nfe_exact": all(
            result["reference_nfe"] == 256 for result in results.values()
        ),
        "std_ratio_in_range": 0.98
        <= results["random"]["diversity_ratios"]["mean_element_std"]
        <= 1.02,
        "pairwise_ratio_in_range": 0.98
        <= results["random"]["diversity_ratios"]["mean_pairwise_distance"]
        <= 1.02,
    }

    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "task": args.task,
        "seed": args.seed,
        "num_observations": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "results": results,
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
