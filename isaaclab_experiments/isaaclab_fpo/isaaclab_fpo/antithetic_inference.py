"""Inference helpers for symmetric flow sources."""

import torch


def _validate_sources(
    observations: torch.Tensor,
    first_source: torch.Tensor,
    second_source: torch.Tensor,
) -> None:
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
    _validate_sources(observations, first_source, second_source)
    kwargs = {
        "integration_method": integration_method,
        "sampling_steps": sampling_steps,
    }
    first = policy.act_inference(observations, source=first_source, **kwargs)
    second = policy.act_inference(observations, source=second_source, **kwargs)
    if first.shape != second.shape:
        raise ValueError("paired endpoint shapes differ")
    return (first + second) * 0.5


def paired_source_action_batched(
    policy,
    observations: torch.Tensor,
    first_source: torch.Tensor,
    second_source: torch.Tensor,
    *,
    integration_method: str | None = None,
    sampling_steps: int | None = None,
) -> torch.Tensor:
    """Average two endpoints computed in one doubled policy batch."""
    _validate_sources(observations, first_source, second_source)
    batch_size = observations.shape[0]
    endpoints = policy.act_inference(
        torch.cat((observations, observations), dim=0),
        source=torch.cat((first_source, second_source), dim=0),
        integration_method=integration_method,
        sampling_steps=sampling_steps,
    )
    if endpoints.shape[0] != 2 * batch_size:
        raise ValueError("batched endpoint output has wrong batch dimension")
    first, second = endpoints.split(batch_size, dim=0)
    if first.shape != second.shape:
        raise ValueError("paired endpoint shapes differ")
    return (first + second) * 0.5


def antithetic_action_batched(
    policy,
    observations: torch.Tensor,
    source: torch.Tensor,
    *,
    integration_method: str | None = None,
    sampling_steps: int | None = None,
) -> torch.Tensor:
    """Average symmetric endpoints computed in one doubled policy batch."""
    return paired_source_action_batched(
        policy,
        observations,
        source,
        -source,
        integration_method=integration_method,
        sampling_steps=sampling_steps,
    )


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
