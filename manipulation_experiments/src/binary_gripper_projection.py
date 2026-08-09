"""Inference-only projection onto the binary gripper command support."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def project_binary_gripper(actions: Tensor) -> Tensor:
    """Return actions with only the final coordinate projected to {-1, +1}."""
    if actions.ndim < 1 or actions.shape[-1] < 1:
        raise ValueError("actions must have a nonempty final dimension")
    if not torch.isfinite(actions).all():
        raise ValueError("actions must be finite")

    projected = actions.clone()
    projected[..., -1] = torch.where(
        actions[..., -1] >= 0,
        torch.ones_like(actions[..., -1]),
        -torch.ones_like(actions[..., -1]),
    )
    return projected


def analyze_binary_gripper_projection(
    raw_actions: np.ndarray,
    projected_actions: np.ndarray,
    successes: Sequence[int],
    initial_observation_hashes: Sequence[str],
    control_observation_hashes: Sequence[str],
    *,
    audit_mode: bool,
) -> dict[str, object]:
    """Evaluate H56 projection invariants and its locked reward gate."""
    raw = np.asarray(raw_actions)
    projected = np.asarray(projected_actions)
    outcomes = np.asarray(successes)
    if raw.ndim != 2 or raw.shape[1] < 2 or raw.shape != projected.shape:
        raise ValueError("action arrays must have matching (step, dimension) shapes")
    if raw.shape[0] < 1 or not np.isfinite(raw).all() or not np.isfinite(projected).all():
        raise ValueError("action arrays must be nonempty and finite")
    if outcomes.ndim != 1 or outcomes.size < 1 or not np.isin(outcomes, [0, 1]).all():
        raise ValueError("successes must be a nonempty binary vector")
    if not (
        len(initial_observation_hashes)
        == len(control_observation_hashes)
        == outcomes.size
    ):
        raise ValueError("observation hashes must match the episode count")

    arm_exact = np.array_equal(raw[:, :-1], projected[:, :-1])
    projected_gripper = projected[:, -1]
    support_exact = bool(np.isin(projected_gripper, [-1.0, 1.0]).all())
    absolute_change = np.abs(projected_gripper - raw[:, -1])
    materially_changed = int((absolute_change > 0.05).sum())
    hashes_exact = list(initial_observation_hashes) == list(control_observation_hashes)
    validity_passed = bool(
        arm_exact and support_exact and materially_changed > 0 and hashes_exact
    )
    success_count = int(outcomes.sum())
    return {
        "episode_count": int(outcomes.size),
        "successes": success_count,
        "success_rate": success_count / int(outcomes.size),
        "executed_action_count": int(raw.shape[0]),
        "arm_coordinates_bitwise_exact": arm_exact,
        "projected_gripper_support_exact": support_exact,
        "materially_changed_gripper_count": materially_changed,
        "materially_changed_gripper_fraction": materially_changed / int(raw.shape[0]),
        "mean_absolute_gripper_change": float(absolute_change.mean()),
        "max_absolute_gripper_change": float(absolute_change.max()),
        "raw_gripper_min": float(raw[:, -1].min()),
        "raw_gripper_max": float(raw[:, -1].max()),
        "initial_observation_hashes_exact": hashes_exact,
        "validity_passed": validity_passed,
        "reward_gate": 17 if audit_mode else None,
        "passed": bool(validity_passed and (not audit_mode or success_count >= 17)),
    }
