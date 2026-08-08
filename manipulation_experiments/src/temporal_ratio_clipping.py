from __future__ import annotations

import torch


def clipped_ratio_objective(
    old_losses: torch.Tensor,
    current_losses: torch.Tensor,
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_coef: float,
    *,
    temporal: bool,
    clamp_logratio: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ratios and a chunk-scale PPO loss for fixed CFM samples."""
    if old_losses.shape != current_losses.shape or old_losses.ndim != 3:
        raise ValueError("losses must share shape [batch, time, samples]")
    if valid_mask.shape != old_losses.shape[:2]:
        raise ValueError("valid_mask must have shape [batch, time]")
    if advantages.shape != old_losses.shape[:1]:
        raise ValueError("advantages must have shape [batch]")

    valid = valid_mask.to(dtype=old_losses.dtype)
    weights = advantages[:, None, None]
    if temporal:
        logratio = (old_losses - current_losses) * valid[:, :, None]
        if clamp_logratio is not None:
            logratio = logratio + (
                logratio.clamp(-clamp_logratio, clamp_logratio) - logratio
            ).detach()
        ratios = logratio.exp()
        unclipped = -weights * ratios
        clipped = -weights * ratios.clamp(1.0 - clip_coef, 1.0 + clip_coef)
        per_timestep = torch.maximum(unclipped, clipped) * valid[:, :, None]
        loss = per_timestep.sum(dim=1).mean()
    else:
        old_chunk = (old_losses * valid[:, :, None]).sum(dim=1)
        current_chunk = (current_losses * valid[:, :, None]).sum(dim=1)
        logratio = old_chunk - current_chunk
        if clamp_logratio is not None:
            logratio = logratio + (
                logratio.clamp(-clamp_logratio, clamp_logratio) - logratio
            ).detach()
        ratios = logratio.exp()
        unclipped = -advantages[:, None] * ratios
        clipped = -advantages[:, None] * ratios.clamp(
            1.0 - clip_coef, 1.0 + clip_coef
        )
        loss = torch.maximum(unclipped, clipped).mean()
    return ratios, loss


def positive_active_fraction(
    ratios: torch.Tensor,
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_coef: float,
) -> torch.Tensor:
    """Measure positive-advantage ratios still below PPO's upper boundary."""
    positive = advantages > 0
    if ratios.ndim == 2:
        selected = ratios[positive]
    elif ratios.ndim == 3:
        selected = ratios[positive][valid_mask[positive].bool()]
    else:
        raise ValueError("ratios must have shape [batch, samples] or [batch, time, samples]")
    if selected.numel() == 0:
        raise ValueError("positive active fraction requires positive valid samples")
    return (selected < 1.0 + clip_coef).float().mean()
