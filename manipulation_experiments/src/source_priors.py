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


def split_action_history(
    action_sequence: Tensor,
    action_is_pad: Tensor,
    history_steps: int,
    horizon: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split a dataset action query into previous actions and current targets."""
    expected_steps = history_steps + horizon
    if action_sequence.ndim != 3:
        raise ValueError("action_sequence must have shape (batch, time, action_dim)")
    if action_sequence.shape[1] != expected_steps:
        raise ValueError(f"expected {expected_steps} action positions, got {action_sequence.shape[1]}")
    if action_is_pad.shape != action_sequence.shape[:2]:
        raise ValueError("action_is_pad must match the batch and time dimensions")

    previous_actions = action_sequence[:, :history_steps]
    target_actions = action_sequence[:, history_steps:]
    target_is_pad = action_is_pad[:, history_steps:]
    has_previous_actions = ~action_is_pad[:, :history_steps].to(torch.bool).any(dim=1)
    return previous_actions, target_actions, target_is_pad, has_previous_actions
