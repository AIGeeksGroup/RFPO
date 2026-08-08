"""Geometry metrics for discretized conditional flow trajectories."""

import torch
from torch import Tensor


def flow_geometry(source: Tensor, path: Tensor, eps: float = 1e-8) -> dict[str, Tensor]:
    """Return per-sample straightness error and path-length ratio.

    `source` has shape (B, T, D). `path` has shape (B, S, T, D) and contains
    the state after each of S equal Euler steps, including the final endpoint.
    """
    if source.ndim != 3 or path.ndim != 4:
        raise ValueError("source and path must have shapes (B,T,D) and (B,S,T,D)")
    if path.shape[0] != source.shape[0] or path.shape[2:] != source.shape[1:]:
        raise ValueError("source and path batch/action dimensions must match")
    if path.shape[1] < 1:
        raise ValueError("path must contain at least one integration step")

    endpoint = path[:, -1]
    alpha = torch.linspace(
        1.0 / path.shape[1], 1.0, path.shape[1], device=path.device, dtype=path.dtype
    ).view(1, -1, 1, 1)
    straight_path = source[:, None] + alpha * (endpoint - source)[:, None]

    displacement_sq = (endpoint - source).square().flatten(1).sum(dim=1)
    deviation_sq = (path - straight_path).square().flatten(2).sum(dim=2).mean(dim=1)
    straightness_error = deviation_sq / displacement_sq.clamp_min(eps)

    full_path = torch.cat((source[:, None], path), dim=1)
    segment_lengths = (full_path[:, 1:] - full_path[:, :-1]).flatten(2).norm(dim=2)
    endpoint_distance = (endpoint - source).flatten(1).norm(dim=1)
    path_length_ratio = segment_lengths.sum(dim=1) / endpoint_distance.clamp_min(eps)

    return {
        "straightness_error": straightness_error,
        "path_length_ratio": path_length_ratio,
    }
