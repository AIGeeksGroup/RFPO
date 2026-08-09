"""Coordinate-wise hybridization of stochastic arm and deterministic gripper outputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from src.factorized_source import tensor_sha256


def substitute_deterministic_gripper(
    gaussian_actions: Tensor, deterministic_actions: Tensor
) -> Tensor:
    """Keep Gaussian arm coordinates and substitute the deterministic gripper."""
    if gaussian_actions.shape != deterministic_actions.shape or gaussian_actions.ndim != 3:
        raise ValueError("action chunks must have matching (batch, horizon, dimension) shapes")
    if gaussian_actions.shape[-1] < 2:
        raise ValueError("action chunks must include arm and gripper coordinates")
    if not torch.isfinite(gaussian_actions).all() or not torch.isfinite(deterministic_actions).all():
        raise ValueError("action chunks must be finite")
    hybrid = gaussian_actions.clone()
    hybrid[..., -1] = deterministic_actions[..., -1]
    return hybrid


def build_hybrid_output_audit_records(
    gaussian_actions: Tensor,
    deterministic_actions: Tensor,
    hybrid_actions: Tensor,
    environment_ids: Sequence[int],
    plan_indices: Sequence[int],
) -> list[dict[str, object]]:
    if not (
        gaussian_actions.shape == deterministic_actions.shape == hybrid_actions.shape
    ) or gaussian_actions.ndim != 3:
        raise ValueError("all action chunks must have matching three-dimensional shapes")
    if gaussian_actions.shape[0] != len(environment_ids) or len(environment_ids) != len(plan_indices):
        raise ValueError("action batch and key lengths must match")

    records = []
    for row, (environment_id, plan_index) in enumerate(
        zip(environment_ids, plan_indices)
    ):
        gaussian_arm = gaussian_actions[row, ..., :-1]
        hybrid_arm = hybrid_actions[row, ..., :-1]
        deterministic_gripper = deterministic_actions[row, ..., -1]
        hybrid_gripper = hybrid_actions[row, ..., -1]
        records.append(
            {
                "environment_id": int(environment_id),
                "plan_index": int(plan_index),
                "gaussian_arm_sha256": tensor_sha256(gaussian_arm),
                "hybrid_arm_sha256": tensor_sha256(hybrid_arm),
                "deterministic_gripper_sha256": tensor_sha256(
                    deterministic_gripper
                ),
                "hybrid_gripper_sha256": tensor_sha256(hybrid_gripper),
                "arm_output_bitwise_exact": bool(
                    torch.equal(gaussian_arm, hybrid_arm)
                ),
                "gripper_output_bitwise_exact": bool(
                    torch.equal(deterministic_gripper, hybrid_gripper)
                ),
                "gaussian_deterministic_gripper_mean_abs": float(
                    (gaussian_actions[row, ..., -1] - deterministic_gripper)
                    .abs()
                    .mean()
                    .item()
                ),
            }
        )
    return records


def analyze_hybrid_gripper_audit(
    control: dict[str, object],
    candidate: dict[str, object],
    *,
    audit_mode: bool,
) -> dict[str, object]:
    """Evaluate H58 common-state/source, output substitution, and reward gates."""
    control_episodes = control["episodes"]
    candidate_episodes = candidate["episodes"]
    if not isinstance(control_episodes, list) or not isinstance(candidate_episodes, list):
        raise ValueError("episodes must be lists")
    if len(control_episodes) != len(candidate_episodes) or not control_episodes:
        raise ValueError("conditions must contain equal nonzero episode counts")
    if [episode["environment_seed"] for episode in control_episodes] != [
        episode["environment_seed"] for episode in candidate_episodes
    ]:
        raise ValueError("condition environment seeds must match")

    control_sources = {
        (record["environment_id"], record["plan_index"]): record
        for record in control["source_records"]
    }
    candidate_sources = {
        (record["environment_id"], record["plan_index"]): record
        for record in candidate["source_records"]
    }
    common_keys = sorted(control_sources.keys() & candidate_sources.keys())
    if not common_keys:
        raise ValueError("conditions must have common source keys")
    arm_source_hashes_exact = all(
        control_sources[key]["used_arm_sha256"]
        == candidate_sources[key]["used_arm_sha256"]
        == control_sources[key]["raw_arm_sha256"]
        == candidate_sources[key]["raw_arm_sha256"]
        for key in common_keys
    )
    hybrid_records = candidate["hybrid_output_records"]
    if not isinstance(hybrid_records, list) or not hybrid_records:
        raise ValueError("candidate must include hybrid output records")
    arm_output_exact = all(
        record["arm_output_bitwise_exact"] for record in hybrid_records
    )
    gripper_output_exact = all(
        record["gripper_output_bitwise_exact"] for record in hybrid_records
    )
    active_records = sum(
        record["gaussian_deterministic_gripper_mean_abs"] > 1e-6
        for record in hybrid_records
    )
    initial_hashes_exact = (
        control["initial_observation_hashes"]
        == candidate["initial_observation_hashes"]
    )
    control_first = np.asarray(control["first_actions"])
    candidate_first = np.asarray(candidate["first_actions"])
    if control_first.shape != candidate_first.shape:
        raise ValueError("condition first actions must have matching shapes")
    first_action_difference = float(np.abs(control_first - candidate_first).max())
    control_successes = int(sum(episode["success"] for episode in control_episodes))
    candidate_successes = int(sum(episode["success"] for episode in candidate_episodes))
    expected_control = not audit_mode or control_successes == 1
    validity_passed = bool(
        arm_source_hashes_exact
        and arm_output_exact
        and gripper_output_exact
        and active_records > 0
        and initial_hashes_exact
        and first_action_difference > 1e-6
        and expected_control
    )
    gain = candidate_successes - control_successes
    return {
        "episode_count_per_condition": len(control_episodes),
        "control_successes": control_successes,
        "candidate_successes": candidate_successes,
        "success_gain": gain,
        "common_source_key_count": len(common_keys),
        "arm_source_hashes_exact": arm_source_hashes_exact,
        "hybrid_record_count": len(hybrid_records),
        "active_hybrid_record_count": int(active_records),
        "arm_output_bitwise_exact": arm_output_exact,
        "gripper_output_bitwise_exact": gripper_output_exact,
        "initial_observation_hashes_exact": initial_hashes_exact,
        "first_action_difference_max_abs": first_action_difference,
        "expected_control_reproduced": expected_control,
        "validity_passed": validity_passed,
        "reward_gain_gate": 3 if audit_mode else None,
        "passed": bool(validity_passed and (not audit_mode or gain >= 3)),
    }
