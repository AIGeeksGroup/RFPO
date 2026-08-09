"""Stateless common-random sources for hybrid arm/gripper flow policies."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def _derived_seed(base_seed: int, environment_id: int, plan_index: int) -> int:
    if base_seed < 0 or environment_id < 0 or plan_index < 0:
        raise ValueError("source seed keys must be non-negative")
    return (base_seed + environment_id * 1_000_003 + plan_index * 10_000_019) % (
        2**63 - 1
    )


def build_stateless_gaussian_sources(
    *,
    base_seed: int,
    environment_ids: Sequence[int],
    plan_indices: Sequence[int],
    horizon: int,
    action_dim: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate sources keyed independently by environment and local plan index."""
    if len(environment_ids) != len(plan_indices) or len(environment_ids) < 1:
        raise ValueError("environment_ids and plan_indices must be nonempty and aligned")
    if horizon < 1 or action_dim < 2:
        raise ValueError("source horizon must be positive and action_dim must include arm/gripper")
    if not dtype.is_floating_point:
        raise ValueError("source dtype must be floating point")

    sources = []
    for environment_id, plan_index in zip(environment_ids, plan_indices):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_derived_seed(base_seed, environment_id, plan_index))
        source = torch.randn(
            (horizon, action_dim),
            generator=generator,
            device="cpu",
            dtype=dtype,
        )
        sources.append(source)
    return torch.stack(sources).to(device=device)


def zero_gripper_source(source: Tensor) -> Tensor:
    """Zero only the final latent coordinate without mutating the source."""
    if source.ndim != 3 or source.shape[-1] < 2:
        raise ValueError("source must have shape (batch, horizon, action_dim>=2)")
    if not torch.isfinite(source).all():
        raise ValueError("source must be finite")
    factorized = source.clone()
    factorized[..., -1] = 0
    return factorized


def tensor_sha256(value: Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_source_audit_records(
    raw_source: Tensor,
    used_source: Tensor,
    environment_ids: Sequence[int],
    plan_indices: Sequence[int],
) -> list[dict[str, object]]:
    if raw_source.shape != used_source.shape or raw_source.ndim != 3:
        raise ValueError("raw and used sources must have matching three-dimensional shapes")
    if raw_source.shape[0] != len(environment_ids) or len(environment_ids) != len(plan_indices):
        raise ValueError("source batch and key lengths must match")

    records = []
    for row, (environment_id, plan_index) in enumerate(
        zip(environment_ids, plan_indices)
    ):
        raw_arm = raw_source[row, ..., :-1]
        used_arm = used_source[row, ..., :-1]
        records.append(
            {
                "environment_id": int(environment_id),
                "plan_index": int(plan_index),
                "raw_arm_sha256": tensor_sha256(raw_arm),
                "used_arm_sha256": tensor_sha256(used_arm),
                "raw_gripper_sha256": tensor_sha256(raw_source[row, ..., -1]),
                "used_gripper_sha256": tensor_sha256(used_source[row, ..., -1]),
                "used_gripper_max_abs": float(
                    used_source[row, ..., -1].abs().max().item()
                ),
                "raw_arm_mean": float(raw_arm.mean().item()),
                "raw_arm_std": float(raw_arm.std(unbiased=False).item()),
            }
        )
    return records


def analyze_factorized_source_audit(
    control: dict[str, object],
    candidate: dict[str, object],
    *,
    audit_mode: bool,
) -> dict[str, object]:
    """Evaluate H57 common-state, common-source, activity, and reward gates."""
    control_episodes = control["episodes"]
    candidate_episodes = candidate["episodes"]
    if not isinstance(control_episodes, list) or not isinstance(candidate_episodes, list):
        raise ValueError("episodes must be lists")
    if len(control_episodes) != len(candidate_episodes) or not control_episodes:
        raise ValueError("conditions must contain equal nonzero episode counts")
    control_seeds = [episode["environment_seed"] for episode in control_episodes]
    candidate_seeds = [episode["environment_seed"] for episode in candidate_episodes]
    if control_seeds != candidate_seeds:
        raise ValueError("condition environment seeds must match")

    control_records = {
        (record["environment_id"], record["plan_index"]): record
        for record in control["source_records"]
    }
    candidate_records = {
        (record["environment_id"], record["plan_index"]): record
        for record in candidate["source_records"]
    }
    common_keys = sorted(control_records.keys() & candidate_records.keys())
    if not common_keys:
        raise ValueError("conditions must have common source keys")
    arm_hashes_exact = all(
        control_records[key]["used_arm_sha256"]
        == candidate_records[key]["used_arm_sha256"]
        == control_records[key]["raw_arm_sha256"]
        == candidate_records[key]["raw_arm_sha256"]
        for key in common_keys
    )
    candidate_gripper_zero = all(
        record["used_gripper_max_abs"] == 0.0
        for record in candidate_records.values()
    )
    control_gripper_active = any(
        record["used_gripper_max_abs"] > 0.0
        for record in control_records.values()
    )
    initial_hashes_exact = (
        control["initial_observation_hashes"]
        == candidate["initial_observation_hashes"]
    )
    control_first = np.asarray(control["first_actions"])
    candidate_first = np.asarray(candidate["first_actions"])
    if control_first.shape != candidate_first.shape or not np.isfinite(control_first).all() or not np.isfinite(candidate_first).all():
        raise ValueError("first actions must be matching finite arrays")
    action_difference_max_abs = float(np.abs(control_first - candidate_first).max())
    control_successes = int(sum(episode["success"] for episode in control_episodes))
    candidate_successes = int(sum(episode["success"] for episode in candidate_episodes))
    control_regime_valid = not audit_mode or 1 <= control_successes <= 8
    validity_passed = bool(
        arm_hashes_exact
        and candidate_gripper_zero
        and control_gripper_active
        and initial_hashes_exact
        and action_difference_max_abs > 1e-6
        and control_regime_valid
    )
    gain = candidate_successes - control_successes
    return {
        "episode_count_per_condition": len(control_episodes),
        "control_successes": control_successes,
        "candidate_successes": candidate_successes,
        "success_gain": gain,
        "common_source_key_count": len(common_keys),
        "arm_source_hashes_exact": arm_hashes_exact,
        "candidate_gripper_source_zero": candidate_gripper_zero,
        "control_gripper_source_active": control_gripper_active,
        "initial_observation_hashes_exact": initial_hashes_exact,
        "first_action_difference_max_abs": action_difference_max_abs,
        "control_regime_valid": control_regime_valid,
        "validity_passed": validity_passed,
        "reward_gain_gate": 3 if audit_mode else None,
        "passed": bool(validity_passed and (not audit_mode or gain >= 3)),
    }
