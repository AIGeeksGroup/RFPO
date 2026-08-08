from __future__ import annotations

import torch


def build_rollout_zero_sampling_mask(
    num_local_envs: int,
    zero_fraction: float,
    *,
    global_num_envs: int | None = None,
    global_offset: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Assign a deterministic global subset of rollout environments to zero sampling."""
    if not 0.0 <= zero_fraction <= 1.0:
        raise ValueError(f"zero_fraction must be in [0, 1], got {zero_fraction}")
    if num_local_envs < 0:
        raise ValueError(f"num_local_envs must be non-negative, got {num_local_envs}")

    total_envs = num_local_envs if global_num_envs is None else global_num_envs
    if total_envs < num_local_envs or global_offset < 0 or global_offset + num_local_envs > total_envs:
        raise ValueError("local environment range must fit within global_num_envs")

    num_zero_envs = int(total_envs * zero_fraction + 0.5)
    global_ids = torch.arange(global_offset, global_offset + num_local_envs, device=device)
    return global_ids < num_zero_envs


def apply_zero_sampling_mask(source_noise: torch.Tensor, zero_sampling_mask: torch.Tensor) -> torch.Tensor:
    """Return source noise with selected batch elements replaced by the zero source."""
    if zero_sampling_mask.dtype != torch.bool:
        raise ValueError("zero_sampling_mask must have boolean dtype")
    if zero_sampling_mask.ndim != 1 or zero_sampling_mask.shape[0] != source_noise.shape[0]:
        raise ValueError(
            "zero_sampling_mask must have shape "
            f"({source_noise.shape[0]},), got {tuple(zero_sampling_mask.shape)}"
        )

    mixed_source = source_noise.clone()
    mixed_source[zero_sampling_mask.to(device=source_noise.device)] = 0
    return mixed_source


def apply_source_sampling_scale(source_noise: torch.Tensor, source_sampling_scale: torch.Tensor) -> torch.Tensor:
    """Scale each batch element's source noise without mutating the input."""
    if source_sampling_scale.ndim != 1 or source_sampling_scale.shape[0] != source_noise.shape[0]:
        raise ValueError(
            "source_sampling_scale must have shape "
            f"({source_noise.shape[0]},), got {tuple(source_sampling_scale.shape)}"
        )
    if (source_sampling_scale < 0).any():
        raise ValueError("source_sampling_scale values must be non-negative")

    scale = source_sampling_scale.to(device=source_noise.device, dtype=source_noise.dtype)
    return source_noise * scale.reshape(-1, *([1] * (source_noise.ndim - 1)))


def prepare_invalid_step_mask(
    invalid_steps: torch.Tensor,
    *,
    reset_each_iteration: bool,
) -> None:
    """Prepare a reused invalid-step buffer for a new rollout iteration."""
    if reset_each_iteration:
        invalid_steps.zero_()
