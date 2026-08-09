"""Exact fresh-state AdamW steps for the preregistered extragradient audit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


@torch.no_grad()
def snapshot_parameters(parameters: Iterable[nn.Parameter]) -> tuple[Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in parameters)


@torch.no_grad()
def restore_parameters(
    parameters: Sequence[nn.Parameter], snapshot: Sequence[Tensor]
) -> None:
    if len(parameters) != len(snapshot):
        raise ValueError("parameter and snapshot lengths differ")
    for parameter, value in zip(parameters, snapshot, strict=True):
        if parameter.shape != value.shape:
            raise ValueError("parameter and snapshot shapes differ")
        parameter.copy_(value)


def parameters_equal_snapshot(
    parameters: Sequence[nn.Parameter], snapshot: Sequence[Tensor]
) -> bool:
    return len(parameters) == len(snapshot) and all(
        torch.equal(parameter.detach(), value)
        for parameter, value in zip(parameters, snapshot, strict=True)
    )


@torch.no_grad()
def assign_flat_gradient(
    parameters: Sequence[nn.Parameter], gradient: Tensor
) -> None:
    expected = sum(parameter.numel() for parameter in parameters)
    if gradient.numel() != expected:
        raise ValueError(
            f"flat gradient has {gradient.numel()} elements; expected {expected}"
        )
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        value = gradient[offset : offset + count].reshape_as(parameter)
        parameter.grad = value.to(device=parameter.device, dtype=parameter.dtype).clone()
        offset += count


@torch.no_grad()
def displacement_vector(
    parameters: Sequence[nn.Parameter], snapshot: Sequence[Tensor]
) -> Tensor:
    if len(parameters) != len(snapshot):
        raise ValueError("parameter and snapshot lengths differ")
    return torch.cat(
        [
            (parameter.detach() - value.to(parameter.device)).flatten().cpu()
            for parameter, value in zip(parameters, snapshot, strict=True)
        ]
    )


def fresh_adamw_step(
    parameters: Sequence[nn.Parameter],
    gradient: Tensor,
    *,
    learning_rate: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    max_grad_norm: float,
) -> dict[str, float]:
    """Apply one AdamW step from empty optimizer state using a flat gradient."""
    if not parameters:
        raise ValueError("at least one trainable parameter is required")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    assign_flat_gradient(parameters, gradient)
    norm_before = nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    norm_after = nn.utils.clip_grad_norm_(parameters, float("inf"))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "gradient_norm_before_clip": float(norm_before.item()),
        "gradient_norm_after_clip": float(norm_after.item()),
    }
