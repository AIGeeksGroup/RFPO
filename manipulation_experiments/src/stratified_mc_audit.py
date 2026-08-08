"""Helpers for fixed-budget advantage-stratified gradient audits."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def advantage_stratified_sample_counts(
    advantages: torch.Tensor,
    *,
    low_samples: int = 4,
    high_samples: int = 12,
) -> torch.Tensor:
    """Assign the larger MC budget to the upper half of absolute advantages."""
    if advantages.ndim != 1 or advantages.numel() == 0 or advantages.numel() % 2:
        raise ValueError("advantages must be a nonempty, even-length vector")
    if low_samples < 1 or high_samples <= low_samples:
        raise ValueError("sample counts must satisfy 1 <= low_samples < high_samples")

    counts = torch.full_like(advantages, low_samples, dtype=torch.long)
    order = torch.argsort(advantages.abs(), descending=True, stable=True)
    counts[order[: advantages.numel() // 2]] = high_samples
    return counts


def gradient_estimator_metrics(
    gradients: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    """Summarize repeated gradient estimates against a fixed reference vector."""
    if gradients.ndim != 2 or reference.ndim != 1:
        raise ValueError("gradients must be 2D and reference must be 1D")
    if gradients.shape[1] != reference.numel() or gradients.shape[0] < 2:
        raise ValueError("gradient shapes are incompatible or fewer than two repeats")
    if not torch.isfinite(gradients).all() or not torch.isfinite(reference).all():
        raise ValueError("gradients must be finite")
    reference_norm_sq = reference.square().sum()
    if reference_norm_sq <= 0 or (gradients.norm(dim=1) <= 0).any():
        raise ValueError("gradients must be nonzero")

    errors = (gradients - reference).square().sum(dim=1) / reference_norm_sq
    cosines = F.cosine_similarity(gradients, reference.unsqueeze(0), dim=1)
    average = gradients.mean(dim=0)
    return {
        "mean_normalized_gradient_mse": errors.mean().item(),
        "median_normalized_gradient_mse": errors.median().item(),
        "mean_gradient_cosine_to_reference": cosines.mean().item(),
        "repeat_average_gradient_norm": average.norm().item(),
    }
