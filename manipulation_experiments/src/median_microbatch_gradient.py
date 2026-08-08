from __future__ import annotations

import torch


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
