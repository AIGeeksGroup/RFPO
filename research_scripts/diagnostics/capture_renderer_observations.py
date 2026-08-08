"""Capture deterministic observations for comparing MuJoCo render backends."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.dexmg_env import create_vectorized_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--state-input", type=Path)
    return parser.parse_args()


def to_numpy(observations: dict[str, torch.Tensor], prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.{key}": value.detach().cpu().numpy()
        for key, value in observations.items()
        if isinstance(value, torch.Tensor)
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = create_vectorized_env(
        env_name="Can",
        num_envs=1,
        device="cpu",
        camera_size=84,
        video_key="agentview",
        expected_image_keys=["robot0_eye_in_hand"],
        seeds=[args.seed],
        debug=True,
    )
    try:
        start = time.perf_counter()
        reset_observations, _ = env.reset(seed=args.seed)
        reset_seconds = time.perf_counter() - start

        wrapped_env = env.vec_env.envs[0]
        robosuite_env = wrapped_env.env
        if args.state_input is not None:
            reference = np.load(args.state_input)
            robosuite_env.sim.set_state_from_flattened(reference["sim_state"])
            robosuite_env.sim.forward()
            raw_observations = robosuite_env._get_observations(force_update=True)
            processed = wrapped_env._process_obs(raw_observations)
            reset_observations = {
                key: torch.from_numpy(value[None]) for key, value in processed.items()
            }

        sim_state = robosuite_env.sim.get_state().flatten()

        start = time.perf_counter()
        step_observations, *_ = env.step(torch.zeros((1, 7)))
        step_seconds = time.perf_counter() - start

        arrays = {
            **to_numpy(reset_observations, "reset"),
            **to_numpy(step_observations, "step"),
            "timing.seconds": np.asarray([reset_seconds, step_seconds]),
            "seed": np.asarray(args.seed),
            "sim_state": sim_state,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        print(f"saved={args.output}", flush=True)
        print(f"reset_seconds={reset_seconds:.4f} step_seconds={step_seconds:.4f}", flush=True)
        for key, value in arrays.items():
            if key.startswith(("reset.", "step.")):
                print(f"{key} shape={value.shape} dtype={value.dtype}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
