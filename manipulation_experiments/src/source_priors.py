"""Source-distribution helpers for flow-policy sampling."""

import torch
from torch import Tensor


def apply_previous_action_prior(
    source_noise: Tensor,
    previous_actions: Tensor,
    has_previous_actions: Tensor,
    sigma: float,
) -> Tensor:
    """Warm the reliable prefix of a flow source with previous actions.

    Environments without a complete previous chunk retain their original source.
    The remaining prediction horizon is also left unchanged.
    """
    if source_noise.ndim != 3 or previous_actions.ndim != 3:
        raise ValueError("source_noise and previous_actions must have shape (batch, time, action_dim)")
    if source_noise.shape[0] != previous_actions.shape[0]:
        raise ValueError("source_noise and previous_actions must have the same batch size")
    if source_noise.shape[2] != previous_actions.shape[2]:
        raise ValueError("source_noise and previous_actions must have the same action dimension")
    if previous_actions.shape[1] > source_noise.shape[1]:
        raise ValueError("previous action chunk cannot exceed the prediction horizon")
    if has_previous_actions.shape != (source_noise.shape[0],):
        raise ValueError("has_previous_actions must have shape (batch,)")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    warm_steps = previous_actions.shape[1]
    warm_mask = has_previous_actions.to(device=source_noise.device, dtype=torch.bool).view(-1, 1, 1)
    warm_source = previous_actions + sigma * source_noise[:, :warm_steps]

    source = source_noise.clone()
    source[:, :warm_steps] = torch.where(warm_mask, warm_source, source[:, :warm_steps])
    return source
