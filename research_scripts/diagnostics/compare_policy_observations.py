"""Measure policy sensitivity to observations captured by two render backends."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from eval_checkpoint import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_observation(path: Path, device: str) -> dict[str, torch.Tensor]:
    archive = np.load(path)
    prefix = "reset."
    return {
        key.removeprefix(prefix): torch.from_numpy(archive[key]).to(device)
        for key in archive.files
        if key.startswith(prefix)
    }


def select_action(policy, observation: dict[str, torch.Tensor]) -> torch.Tensor:
    policy.reset()
    torch.manual_seed(0)
    with torch.inference_mode():
        action, _ = policy.select_action(observation, zero_sampling=True)
    return action.detach().cpu()


def main() -> None:
    args = parse_args()
    policy = load_policy(args.checkpoint, device=args.device, load_ema=True)
    policy.init_action_buffers(1)
    policy.eval()

    reference_action = select_action(policy, load_observation(args.reference, args.device))
    candidate_action = select_action(policy, load_observation(args.candidate, args.device))
    difference = (reference_action - candidate_action).abs()

    print(f"reference_action={reference_action.numpy().tolist()}")
    print(f"candidate_action={candidate_action.numpy().tolist()}")
    print(f"max_abs_difference={difference.max().item():.8f}")
    print(f"mean_abs_difference={difference.mean().item():.8f}")
    print(f"reference_action_l2={reference_action.norm().item():.8f}")
    print(f"difference_l2={difference.norm().item():.8f}")


if __name__ == "__main__":
    main()
