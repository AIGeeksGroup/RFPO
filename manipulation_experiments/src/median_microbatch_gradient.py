from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GeometricMedianResult:
    value: torch.Tensor
    iterations: int
    converged: bool


def middle_pair_mean(values: torch.Tensor, *, dim: int = 0) -> torch.Tensor:
    """Return the mean of the two middle values along an even-sized dimension."""
    if values.ndim == 0 or values.shape[dim] < 2 or values.shape[dim] % 2:
        raise ValueError("middle-pair mean requires a positive even-sized dimension")
    ordered = values.sort(dim=dim).values
    upper = values.shape[dim] // 2
    return 0.5 * (
        ordered.select(dim, upper - 1) + ordered.select(dim, upper)
    )


def aggregate_microbatch_gradients(
    gradients: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return arithmetic-mean control and coordinate-median candidate vectors."""
    if len(gradients) != 4:
        raise ValueError("median audit requires exactly four microbatch gradients")
    if any(gradient.ndim != 1 for gradient in gradients):
        raise ValueError("microbatch gradients must be flat vectors")
    if any(gradient.shape != gradients[0].shape for gradient in gradients[1:]):
        raise ValueError("microbatch gradients must have matching shapes")
    stacked = torch.stack(gradients)
    if not torch.isfinite(stacked).all():
        raise ValueError("microbatch gradients must be finite")
    return stacked.mean(dim=0), middle_pair_mean(stacked, dim=0)


def geometric_median(
    gradients: list[torch.Tensor],
    *,
    relative_tolerance: float = 1e-6,
    max_iterations: int = 100,
    zero_distance_epsilon: float = 1e-12,
) -> GeometricMedianResult:
    """Compute the Euclidean geometric median with Weiszfeld iterations."""
    if len(gradients) < 2:
        raise ValueError("geometric median requires at least two gradients")
    if any(gradient.ndim != 1 for gradient in gradients):
        raise ValueError("microbatch gradients must be flat vectors")
    if any(gradient.shape != gradients[0].shape for gradient in gradients[1:]):
        raise ValueError("microbatch gradients must have matching shapes")
    if relative_tolerance <= 0 or max_iterations < 1 or zero_distance_epsilon <= 0:
        raise ValueError("geometric median solver settings must be positive")

    stacked = torch.stack(gradients)
    if not torch.isfinite(stacked).all():
        raise ValueError("microbatch gradients must be finite")

    estimate = stacked.mean(dim=0)
    for iteration in range(1, max_iterations + 1):
        distances = torch.linalg.vector_norm(stacked - estimate, dim=1)
        nearest_distance, nearest_index = distances.min(dim=0)
        if nearest_distance <= zero_distance_epsilon:
            return GeometricMedianResult(
                value=stacked[nearest_index].clone(),
                iterations=iteration - 1,
                converged=True,
            )

        inverse_distances = distances.reciprocal()
        next_estimate = (
            stacked * inverse_distances[:, None]
        ).sum(dim=0) / inverse_distances.sum()
        displacement = torch.linalg.vector_norm(next_estimate - estimate)
        scale = torch.linalg.vector_norm(estimate).clamp_min(1.0)
        estimate = next_estimate
        if displacement <= relative_tolerance * scale:
            return GeometricMedianResult(
                value=estimate,
                iterations=iteration,
                converged=True,
            )

    return GeometricMedianResult(
        value=estimate,
        iterations=max_iterations,
        converged=False,
    )
