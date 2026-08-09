"""Inference helpers for symmetric flow sources."""

import torch


def paired_source_action(
    policy,
    observations: torch.Tensor,
    first_source: torch.Tensor,
    second_source: torch.Tensor,
    *,
    integration_method: str | None = None,
    sampling_steps: int | None = None,
) -> torch.Tensor:
    """Average policy endpoints from two explicit flow sources."""
    expected_shape = (observations.shape[0],)
    if first_source.ndim != 2 or first_source.shape[:1] != expected_shape:
        raise ValueError("sources and observations must have matching batch dimensions")
    if second_source.shape != first_source.shape:
        raise ValueError("paired sources must have identical shapes")
    if not first_source.is_floating_point() or not second_source.is_floating_point():
        raise ValueError("sources must be floating-point tensors")
    if (
        not torch.isfinite(first_source).all()
        or not torch.isfinite(second_source).all()
    ):
        raise ValueError("sources must be finite")
    kwargs = {
        "integration_method": integration_method,
        "sampling_steps": sampling_steps,
    }
    first = policy.act_inference(observations, source=first_source, **kwargs)
    second = policy.act_inference(observations, source=second_source, **kwargs)
    if first.shape != second.shape:
        raise ValueError("paired endpoint shapes differ")
    return (first + second) * 0.5


def antithetic_action(
    policy,
    observations: torch.Tensor,
    source: torch.Tensor,
    *,
    integration_method: str | None = None,
    sampling_steps: int | None = None,
) -> torch.Tensor:
    """Average endpoints from one source and its exact negation."""
    return paired_source_action(
        policy,
        observations,
        source,
        -source,
        integration_method=integration_method,
        sampling_steps=sampling_steps,
    )
