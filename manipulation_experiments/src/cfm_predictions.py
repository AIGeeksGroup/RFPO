from __future__ import annotations

import torch
from torch import Tensor


def flow_endpoint_predictions(
    x_t: Tensor,
    t: Tensor,
    network_output: Tensor,
    *,
    output_parameterization: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert a flow-network output into velocity and straight-path endpoints."""
    if output_parameterization == "u":
        velocity_pred = network_output
        x0_pred = x_t - t * velocity_pred
    elif output_parameterization == "x0":
        x0_pred = network_output
        velocity_pred = (x_t - x0_pred) / torch.clamp(t, min=1e-5)
    else:
        raise ValueError(f"unknown flow output parameterization: {output_parameterization}")
    return velocity_pred, x0_pred, x0_pred + velocity_pred

