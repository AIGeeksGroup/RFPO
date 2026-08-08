from __future__ import annotations

import torch


def terminal_consistent_weights(
    normalized_advantages: torch.Tensor,
    centered_terminal_returns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero labeled advantage weights whose signs contradict terminal outcomes."""
    if normalized_advantages.shape != centered_terminal_returns.shape:
        raise ValueError("advantages and terminal returns must have matching shapes")
    if normalized_advantages.ndim != 1 or normalized_advantages.numel() == 0:
        raise ValueError("terminal-consistency inputs must be nonempty vectors")
    if not torch.isfinite(normalized_advantages).all() or not torch.isfinite(
        centered_terminal_returns
    ).all():
        raise ValueError("terminal-consistency inputs must be finite")
    retained = normalized_advantages * centered_terminal_returns > 0
    return torch.where(retained, normalized_advantages, 0.0), retained
