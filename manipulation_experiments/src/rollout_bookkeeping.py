from __future__ import annotations

import torch


def prepare_invalid_step_mask(
    invalid_steps: torch.Tensor,
    *,
    reset_each_iteration: bool,
) -> None:
    """Prepare a reused invalid-step buffer for a new rollout iteration."""
    if reset_each_iteration:
        invalid_steps.zero_()

