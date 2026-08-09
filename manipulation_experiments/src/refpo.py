"""ReFPO loss helpers."""

from __future__ import annotations

from torch import Tensor


def masked_cfm_mean(losses: Tensor, valid_mask: Tensor) -> Tensor:
    """Average `(batch, horizon, MC)` CFM losses over valid action positions."""
    if losses.ndim != 3 or valid_mask.shape != losses.shape[:2]:
        raise ValueError("losses and valid_mask must have shapes (B, T, M) and (B, T)")
    denominator = valid_mask.sum() * losses.shape[-1]
    if denominator.item() <= 0:
        raise ValueError("at least one CFM action position must be valid")
    return (losses * valid_mask.unsqueeze(-1)).sum() / denominator
