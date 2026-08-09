from __future__ import annotations

import torch
from torch import Tensor

from .flow_metrics import flow_geometry


def select_lower_curvature_candidate(
    sources: Tensor,
    paths: Tensor,
    actions: Tensor,
    *,
    action_steps: int,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Select one complete candidate per batch item by normalized path curvature."""
    if sources.ndim != 4 or paths.ndim != 5 or actions.ndim != 4:
        raise ValueError(
            "sources, paths, and actions must have shapes [B,C,T,D], "
            "[B,C,S,T,D], and [B,C,T,D]"
        )
    batch_size, candidate_count, horizon, action_dim = sources.shape
    if candidate_count != 2:
        raise ValueError("curvature selection requires exactly two candidates")
    if paths.shape[:2] != (batch_size, candidate_count):
        raise ValueError("paths must match source batch and candidate dimensions")
    if paths.shape[3:] != (horizon, action_dim) or actions.shape != sources.shape:
        raise ValueError("source, path, and action dimensions must match")
    if not 1 <= action_steps <= horizon:
        raise ValueError("action_steps must lie within the prediction horizon")
    if not all(torch.isfinite(value).all() for value in (sources, paths, actions)):
        raise ValueError("curvature candidates must be finite")

    flat_sources = sources[:, :, :action_steps].reshape(
        batch_size * candidate_count, action_steps, action_dim
    )
    flat_paths = paths[:, :, :, :action_steps].reshape(
        batch_size * candidate_count,
        paths.shape[2],
        action_steps,
        action_dim,
    )
    scores = flow_geometry(flat_sources, flat_paths)["straightness_error"].reshape(
        batch_size, candidate_count
    )
    if not torch.isfinite(scores).all():
        raise ValueError("curvature scores must be finite")

    selected_indices = scores.argmin(dim=1)
    batch_indices = torch.arange(batch_size, device=scores.device)
    selected_actions = actions[batch_indices, selected_indices]
    selected_paths = paths[batch_indices, selected_indices]
    selected_scores = scores[batch_indices, selected_indices]
    return selected_actions, selected_paths, {
        "scores": scores,
        "selected_indices": selected_indices,
        "selected_scores": selected_scores,
    }
