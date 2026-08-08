from __future__ import annotations

from typing import Literal

import torch


def sample_cfm_variables(
    *,
    batch_size: int,
    num_samples: int,
    horizon: int,
    action_dim: int,
    mode: Literal["iid", "joint_antithetic"],
    time_generator: torch.Generator,
    noise_generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample flattened CFM times and noises for a repeated action batch."""
    if min(batch_size, num_samples, horizon, action_dim) < 1:
        raise ValueError("all dimensions must be positive")
    if mode == "joint_antithetic" and num_samples % 2 != 0:
        raise ValueError("joint antithetic sampling requires an even num_samples")

    samples_to_draw = num_samples if mode == "iid" else num_samples // 2
    times = torch.rand(
        (batch_size, samples_to_draw, 1, 1),
        generator=time_generator,
        device=device,
        dtype=dtype,
    )
    noises = torch.randn(
        (batch_size, samples_to_draw, horizon, action_dim),
        generator=noise_generator,
        device=device,
        dtype=dtype,
    )
    if mode == "joint_antithetic":
        times = torch.cat((times, 1.0 - times), dim=1)
        noises = torch.cat((noises, -noises), dim=1)
    elif mode != "iid":
        raise ValueError(f"unsupported CFM sampling mode: {mode}")

    return times.reshape(batch_size * num_samples, 1, 1), noises.reshape(
        batch_size * num_samples, horizon, action_dim
    )
