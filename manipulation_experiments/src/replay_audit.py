from __future__ import annotations

import torch


def successful_chunk_mask(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    n_action_steps: int,
) -> torch.Tensor:
    """Mark flattened environment chunks that lead to a terminal success."""
    if rewards.ndim != 2 or dones.shape != rewards.shape:
        raise ValueError("rewards and dones must have matching [steps, environments] shapes")
    if n_action_steps < 1 or rewards.shape[0] % n_action_steps != 0:
        raise ValueError("n_action_steps must evenly divide the rollout length")

    successful_steps = torch.zeros_like(rewards, dtype=torch.bool)
    for env_index in range(rewards.shape[1]):
        episode_succeeded = False
        for step in range(rewards.shape[0] - 1, -1, -1):
            if rewards[step, env_index] > 0:
                episode_succeeded = True
            successful_steps[step, env_index] = episode_succeeded
            if dones[step, env_index] > 0:
                episode_succeeded = False

    return (
        successful_steps.reshape(-1, n_action_steps, rewards.shape[1])
        .permute(0, 2, 1)
        .any(dim=-1)
        .reshape(-1)
    )


def effective_sample_fraction(weights: torch.Tensor) -> torch.Tensor:
    """Return importance-weight ESS divided by the number of weights."""
    flat = weights.detach().reshape(-1)
    if flat.numel() == 0 or not torch.isfinite(flat).all() or (flat < 0).any():
        raise ValueError("weights must be a nonempty finite nonnegative tensor")
    denominator = flat.numel() * flat.square().sum()
    if denominator <= 0:
        raise ValueError("at least one weight must be positive")
    return flat.sum().square() / denominator
