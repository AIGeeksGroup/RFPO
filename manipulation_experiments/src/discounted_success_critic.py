"""Utilities for discounted terminal-success critic targets."""

from __future__ import annotations

import torch


def centered_average_rank_scores(values: torch.Tensor) -> torch.Tensor:
    """Map a finite vector to centered [-1, 1] average-rank scores."""
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("values must be a vector with at least two entries")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")

    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    scores = 2.0 * ranks / (values.numel() - 1) - 1.0
    scores = scores - scores.mean()
    output_dtype = values.dtype if values.is_floating_point() else torch.float32
    return scores.to(dtype=output_dtype)


def discounted_returns_to_observed_terminal(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    discount: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return discounted rewards and a mask excluding trailing censored segments."""
    if rewards.shape != terminals.shape or rewards.ndim != 2:
        raise ValueError("rewards and terminals must have matching [time, env] shapes")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0, 1]")
    returns = torch.zeros_like(rewards)
    valid = torch.zeros_like(terminals, dtype=torch.bool)
    running_return = torch.zeros_like(rewards[0])
    reaches_terminal = torch.zeros_like(terminals[0], dtype=torch.bool)
    for step in reversed(range(rewards.shape[0])):
        terminal = terminals[step].bool()
        running_return = torch.where(
            terminal,
            rewards[step],
            rewards[step] + discount * running_return,
        )
        reaches_terminal = terminal | reaches_terminal
        returns[step] = running_return
        valid[step] = reaches_terminal
    return returns, valid


def spearman_rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Spearman correlation with average ranks for ties."""
    if x.ndim != 1 or y.ndim != 1 or x.numel() != y.numel() or x.numel() < 2:
        raise ValueError("x and y must be equal-length vectors with at least two values")

    x_centered = centered_average_rank_scores(x.detach().cpu()).double()
    y_centered = centered_average_rank_scores(y.detach().cpu()).double()
    denominator = x_centered.norm() * y_centered.norm()
    if denominator == 0:
        return float("nan")
    return float(torch.dot(x_centered, y_centered).div(denominator).item())
