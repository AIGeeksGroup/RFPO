"""Inference helpers for symmetric flow sources."""

import torch


def antithetic_action(
    policy,
    observations: torch.Tensor,
    source: torch.Tensor,
    *,
    integration_method: str | None = None,
    sampling_steps: int | None = None,
) -> torch.Tensor:
    """Average endpoints from one source and its exact negation."""
    if source.ndim != 2 or source.shape[0] != observations.shape[0]:
        raise ValueError("source and observations must have matching batch dimensions")
    if not source.is_floating_point() or not torch.isfinite(source).all():
        raise ValueError("source must be a finite floating-point tensor")
    kwargs = {
        "integration_method": integration_method,
        "sampling_steps": sampling_steps,
    }
    positive = policy.act_inference(observations, source=source, **kwargs)
    negative = policy.act_inference(observations, source=-source, **kwargs)
    if positive.shape != negative.shape:
        raise ValueError("antithetic endpoint shapes differ")
    return (positive + negative) * 0.5
