"""Target construction helpers for mixed conditional reflow."""

from __future__ import annotations

import torch
from torch import Tensor


def mix_reflow_targets(
    dataset_actions: Tensor,
    dataset_is_pad: Tensor,
    teacher_actions: Tensor,
    use_teacher: Tensor,
) -> tuple[Tensor, Tensor]:
    """Select teacher or dataset endpoints independently for each batch element."""
    if dataset_actions.shape != teacher_actions.shape:
        raise ValueError("dataset and teacher actions must have matching shapes")
    if dataset_is_pad.shape != dataset_actions.shape[:2]:
        raise ValueError("padding mask must match the action batch and horizon")
    if use_teacher.shape != (dataset_actions.shape[0],):
        raise ValueError("use_teacher must contain one boolean per batch element")

    use_teacher = use_teacher.to(device=dataset_actions.device, dtype=torch.bool)
    actions = torch.where(use_teacher[:, None, None], teacher_actions, dataset_actions)
    is_pad = torch.where(
        use_teacher[:, None],
        torch.zeros_like(dataset_is_pad, dtype=torch.bool),
        dataset_is_pad,
    )
    return actions, is_pad

