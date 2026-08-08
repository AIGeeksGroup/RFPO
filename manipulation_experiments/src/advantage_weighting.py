from __future__ import annotations

import torch


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    """Return the scale-invariant effective sample size of nonnegative weights."""
    flat_weights = weights.reshape(-1)
    return flat_weights.sum().square() / flat_weights.square().sum().clamp_min(1e-12)


def clipped_mirror_ratio_loss(
    ratio: torch.Tensor,
    weights: torch.Tensor,
    clip_coefficient: float,
) -> torch.Tensor:
    """Apply positive mirror weights to the standard PPO clipped-ratio surrogate."""
    unclipped_loss = -weights * ratio
    clipped_loss = -weights * torch.clamp(
        ratio, 1 - clip_coefficient, 1 + clip_coefficient
    )
    return torch.max(unclipped_loss, clipped_loss).mean()


def ess_softmax_weights(
    advantages: torch.Tensor,
    target_ess_fraction: float,
    *,
    min_temperature: float = 1e-3,
    max_temperature: float = 1e3,
    bisection_steps: int = 40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mean-one exponential weights at a requested ESS fraction."""
    if not 0.0 < target_ess_fraction <= 1.0:
        raise ValueError("target_ess_fraction must be in (0, 1]")
    if min_temperature <= 0.0 or max_temperature <= min_temperature:
        raise ValueError("temperatures must satisfy 0 < min_temperature < max_temperature")

    detached = advantages.detach()
    flat_advantages = detached.reshape(-1)
    num_samples = flat_advantages.numel()
    if num_samples == 0:
        raise ValueError("advantages must contain at least one value")

    def weights_at(temperature: torch.Tensor) -> torch.Tensor:
        logits = (flat_advantages - flat_advantages.max()) / temperature
        return torch.softmax(logits, dim=0) * num_samples

    target_ess = max(1.0, target_ess_fraction * num_samples)
    low = detached.new_tensor(min_temperature)
    high = detached.new_tensor(max_temperature)

    if effective_sample_size(weights_at(low)) < target_ess:
        for _ in range(bisection_steps):
            midpoint = (low + high) / 2.0
            if effective_sample_size(weights_at(midpoint)) < target_ess:
                low = midpoint
            else:
                high = midpoint

    temperature = high
    flat_weights = weights_at(temperature)
    ess_fraction = effective_sample_size(flat_weights) / num_samples
    return flat_weights.reshape_as(detached), temperature, ess_fraction
