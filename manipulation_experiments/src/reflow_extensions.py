"""Reflow idea helpers ported from isaaclab_fpo for arm FPO++.

Used by reward-aware reflow, adaptive compute, and theory metrics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_advantage_reflow_weights(
    advantages: torch.Tensor,
    mode: str,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Build per-sample weights for reflow losses.

    Args:
        advantages: shape (B,) or (B, 1)
        mode: ``reward_aware`` or ``fpo_operator``
        threshold: advantage offset for reward_aware
    """
    adv = advantages.reshape(-1)
    if mode == "reward_aware":
        weights = torch.clamp(adv - threshold, min=0.0)
    elif mode == "fpo_operator":
        weights = torch.where(adv > 0.0, adv, torch.zeros_like(adv))
    else:
        raise ValueError(f"Unknown reflow weighting mode: {mode}")

    if weights.sum() <= 1e-8:
        return torch.ones_like(weights) / weights.numel()
    return weights / (weights.sum() + 1e-8)


def compute_path_straightness(
    x0: torch.Tensor,
    x1: torch.Tensor,
    target_velocity: torch.Tensor,
) -> torch.Tensor:
    """Rectified-flow straightness: ||u* - (x0-x1)|| / path_len. Lower is straighter."""
    reduce_dims = tuple(range(x0.ndim - 2, x0.ndim)) if x0.ndim >= 2 else (-1,)
    u_star = x0 - x1
    vel_err = (target_velocity - u_star).pow(2).mean(dim=reduce_dims)
    path_len = (x0 - x1).pow(2).sum(dim=reduce_dims).sqrt().clamp(min=1e-6)
    return vel_err / path_len


def compute_discretization_gap(
    action_full: torch.Tensor,
    action_low: torch.Tensor,
) -> torch.Tensor:
    """Euler discretization error proxy: ||a_N - a_k||."""
    reduce_dims = (
        tuple(range(action_full.ndim - 2, action_full.ndim))
        if action_full.ndim >= 2
        else (-1,)
    )
    return (action_full - action_low).pow(2).mean(dim=reduce_dims).sqrt()


class StepPredictor(nn.Module):
    """Predict discrete flow integration budget from observation conditioning."""

    def __init__(self, num_obs: int, step_bins: list[int], hidden_dim: int = 128):
        super().__init__()
        self.step_bins = list(step_bins)
        self.net = nn.Sequential(
            nn.Linear(num_obs, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, len(self.step_bins)),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

    def predict_steps(self, observations: torch.Tensor) -> torch.Tensor:
        logits = self.forward(observations)
        indices = logits.argmax(dim=-1)
        bins = torch.tensor(self.step_bins, device=observations.device, dtype=torch.long)
        return bins[indices]

    def expected_step_fraction(self, observations: torch.Tensor) -> torch.Tensor:
        logits = self.forward(observations)
        probs = torch.softmax(logits, dim=-1)
        max_steps = float(max(self.step_bins))
        normalized_bins = torch.tensor(
            [s / max_steps for s in self.step_bins],
            device=observations.device,
            dtype=probs.dtype,
        )
        return (probs * normalized_bins).sum(dim=-1)
