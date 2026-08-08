from __future__ import annotations

import torch


def gradient_dot(
    first: list[torch.Tensor | None], second: list[torch.Tensor | None]
) -> torch.Tensor:
    """Return a device-preserving dot product for aligned gradient lists."""
    if len(first) != len(second):
        raise ValueError("gradient lists must have equal length")
    terms = [
        (left * right).sum()
        for left, right in zip(first, second, strict=True)
        if left is not None and right is not None
    ]
    if not terms:
        raise ValueError("gradient lists have no shared tensors")
    return torch.stack(terms).sum()


def gradient_norm(gradients: list[torch.Tensor | None]) -> torch.Tensor:
    terms = [gradient.square().sum() for gradient in gradients if gradient is not None]
    if not terms:
        raise ValueError("gradient list has no tensors")
    return torch.stack(terms).sum().sqrt()


def project_conflicting_gradient(
    primary: list[torch.Tensor | None],
    anchor: list[torch.Tensor | None],
    *,
    eps: float = 1e-12,
) -> tuple[list[torch.Tensor | None], torch.Tensor, bool]:
    """Project the primary gradient off an anchor direction only on conflict."""
    dot = gradient_dot(primary, anchor)
    anchor_norm_sq = gradient_dot(anchor, anchor)
    conflict = bool(dot.item() < 0.0 and anchor_norm_sq.item() > eps)
    if not conflict:
        return [None if grad is None else grad.clone() for grad in primary], dot, False

    coefficient = dot / anchor_norm_sq.clamp_min(eps)
    projected = [
        None
        if grad is None
        else grad - coefficient * (torch.zeros_like(grad) if anchor_grad is None else anchor_grad)
        for grad, anchor_grad in zip(primary, anchor, strict=True)
    ]
    return projected, dot, True


def gradient_cosine(
    first: list[torch.Tensor | None], second: list[torch.Tensor | None], eps: float = 1e-12
) -> torch.Tensor:
    return gradient_dot(first, second) / (
        gradient_norm(first) * gradient_norm(second)
    ).clamp_min(eps)

