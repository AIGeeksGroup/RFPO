"""Benchmark sequential and batched H70 antithetic policy inference."""

# ruff: noqa: E402

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-v0")
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=20261610)
parser.add_argument("--warmup-calls", type=int, default=10)
parser.add_argument("--blocks", type=int, default=5)
parser.add_argument("--calls-per-block", type=int, default=20)
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
from isaaclab_fpo.antithetic_inference import (
    antithetic_action,
    antithetic_action_batched,
)
from isaaclab_fpo.runners import OnPolicyRunner
from isaaclab_fpo.task_cfgs import TASK_CONFIGS


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "block_seconds_per_call": values.tolist(),
        "median_seconds_per_call": float(np.median(values)),
        "iqr_seconds_per_call": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
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
    policy = runner.alg.policy
    actor_before = copy.deepcopy(policy.actor.state_dict())

    obs, _ = env.reset()
    obs = obs.to(runner.device)
    with torch.inference_mode():
        norm_obs = (
            runner.obs_normalizer(obs)
            if runner.cfg.empirical_normalization
            else obs
        )
        generator = torch.Generator(device=runner.device)
        generator.manual_seed(args.seed)
        source = torch.randn(
            args.num_envs,
            policy.num_actions,
            device=runner.device,
            dtype=norm_obs.dtype,
            generator=generator,
        )

        methods = {
            "random64": lambda: policy.act_inference(
                norm_obs,
                source=source,
                integration_method="euler",
                sampling_steps=64,
            ),
            "antithetic32_sequential": lambda: antithetic_action(
                policy,
                norm_obs,
                source,
                integration_method="euler",
                sampling_steps=32,
            ),
            "antithetic32_batched": lambda: antithetic_action_batched(
                policy,
                norm_obs,
                source,
                integration_method="euler",
                sampling_steps=32,
            ),
        }
        outputs = {name: method() for name, method in methods.items()}
        torch.cuda.synchronize()
        for method in methods.values():
            for _ in range(args.warmup_calls):
                method()
            torch.cuda.synchronize()

        peak_memory = {}
        for name, method in methods.items():
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            method()
            torch.cuda.synchronize()
            peak_memory[name] = int(torch.cuda.max_memory_allocated() - baseline)

        timings = {name: [] for name in methods}
        names = list(methods)
        for block in range(args.blocks):
            order = names[block % len(names) :] + names[: block % len(names)]
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.calls_per_block):
                    methods[name]()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                timings[name].append(elapsed / args.calls_per_block)

    summaries = {name: summarize(values) for name, values in timings.items()}
    for name, summary in summaries.items():
        summary["actions_per_second"] = args.num_envs / summary["median_seconds_per_call"]
        summary["peak_incremental_cuda_bytes"] = peak_memory[name]
    sequential = summaries["antithetic32_sequential"]["median_seconds_per_call"]
    batched = summaries["antithetic32_batched"]["median_seconds_per_call"]
    random64 = summaries["random64"]["median_seconds_per_call"]
    difference = outputs["antithetic32_batched"] - outputs["antithetic32_sequential"]
    max_abs_difference = float(difference.abs().max())
    gates = {
        "finite_actions": all(bool(torch.isfinite(value).all()) for value in outputs.values()),
        "batched_sequential_max_abs_at_most_1e_5": max_abs_difference <= 1e-5,
        "batched_speedup_at_least_1_15": sequential / batched >= 1.15,
        "batched_random64_latency_ratio_at_most_1_25": batched / random64 <= 1.25,
        "actor_parameters_unchanged": all(
            torch.equal(value, actor_before[key])
            for key, value in policy.actor.state_dict().items()
        ),
    }
    result = {
        "hypothesis": "H71",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "task": args.task,
        "device_name": torch.cuda.get_device_name(),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "warmup_calls": args.warmup_calls,
        "blocks": args.blocks,
        "calls_per_block": args.calls_per_block,
        "timings": summaries,
        "batched_sequential_max_abs_difference": max_abs_difference,
        "batched_sequential_rms_difference": float(torch.sqrt(torch.mean(difference.double().square()))),
        "batched_over_sequential_speedup": sequential / batched,
        "batched_over_random64_latency_ratio": batched / random64,
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
