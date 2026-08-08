from __future__ import annotations

import torch


def rollback_clipped_ratio_loss(
    ratios: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
    rollback_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the PPO-RB loss and mask of improving out-of-bound elements."""
    if not 0.0 < clip_coef < 1.0:
        raise ValueError("clip_coef must be in (0, 1)")
    if rollback_alpha <= 0.0:
        raise ValueError("rollback_alpha must be positive")

    ratios, advantages = torch.broadcast_tensors(ratios, advantages)
    lower = 1.0 - clip_coef
    upper = 1.0 + clip_coef
    rollback_ratios = torch.where(
        ratios < lower,
        -rollback_alpha * ratios + (1.0 + rollback_alpha) * lower,
        torch.where(
            ratios > upper,
            -rollback_alpha * ratios + (1.0 + rollback_alpha) * upper,
            ratios,
        ),
    )
    loss_unclipped = -advantages * ratios
    loss_rollback = -advantages * rollback_ratios
    active = ((advantages > 0) & (ratios > upper)) | (
        (advantages < 0) & (ratios < lower)
    )
    return torch.maximum(loss_unclipped, loss_rollback).mean(), active
