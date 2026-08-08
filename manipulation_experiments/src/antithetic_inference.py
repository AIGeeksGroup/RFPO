from __future__ import annotations

import torch


def build_antithetic_sources(source_noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an independent source tensor and its exact antithetic counterpart."""
    return source_noise.clone(), -source_noise


def average_antithetic_predictions(
    positive_prediction: torch.Tensor,
    negative_prediction: torch.Tensor,
) -> torch.Tensor:
    """Average matched predictions without allowing accidental broadcasting."""
    if positive_prediction.shape != negative_prediction.shape:
        raise ValueError(
            "antithetic predictions must have identical shapes, got "
            f"{tuple(positive_prediction.shape)} and {tuple(negative_prediction.shape)}"
        )
    return 0.5 * (positive_prediction + negative_prediction)
