from __future__ import annotations

from typing import Any

import torch


def clear_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Clear per-parameter moments while preserving parameter groups and hyperparameters."""
    state_entries_before = len(optimizer.state)
    optimizer.state.clear()
    return {
        "state_entries_before": state_entries_before,
        "state_entries_after": len(optimizer.state),
    }
