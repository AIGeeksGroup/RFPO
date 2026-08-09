from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def leave_one_out_advantages(outcomes: np.ndarray) -> np.ndarray:
    values = np.asarray(outcomes, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("outcomes must have shape [groups, replicas>=2]")
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("outcomes must be finite and binary")
    group_size = values.shape[1]
    return (group_size * values - values.sum(axis=1, keepdims=True)) / (
        group_size - 1
    )


def absolute_weight_ess(weights: np.ndarray) -> float:
    values = np.abs(np.asarray(weights, dtype=np.float64).reshape(-1))
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(values.sum() ** 2 / np.square(values).sum())


def summarize_group_relative_audit(
    records: Sequence[dict[str, object]],
    *,
    expected_group_size: int,
    expected_group_count: int,
    source_mean: float,
    source_std: float,
    first_chunk_pairwise_rms: Sequence[float],
    policy_parameters_bitwise_unchanged: bool,
    formal: bool,
) -> dict[str, object]:
    if expected_group_size < 2 or expected_group_count < 1:
        raise ValueError("expected group dimensions must be positive")
    if len(records) != expected_group_size * expected_group_count:
        raise ValueError("records do not match the expected group dimensions")

    grouped: dict[int, list[dict[str, object]]] = {}
    for record in records:
        root_seed = int(record["root_seed"])
        grouped.setdefault(root_seed, []).append(record)
    if len(grouped) != expected_group_count:
        raise ValueError("records do not contain the expected number of root groups")

    ordered_groups = []
    initial_hashes_exact = True
    privileged_hashes_exact = True
    episodes_complete = True
    outcomes = []
    for root_seed in sorted(grouped):
        group = sorted(grouped[root_seed], key=lambda item: int(item["replica"]))
        if len(group) != expected_group_size or [int(item["replica"]) for item in group] != list(
            range(expected_group_size)
        ):
            raise ValueError(f"root {root_seed} has an invalid replica manifest")
        initial_hashes_exact &= len({str(item["initial_observation_hash"]) for item in group}) == 1
        privileged_hashes_exact &= len({str(item["initial_privileged_state_hash"]) for item in group}) == 1
        group_outcomes = [int(item["success"]) for item in group]
        outcomes.append(group_outcomes)
        episodes_complete &= all(
            bool(item["completed"])
            and 1 <= int(item["length"]) <= 400
            and int(item["success"]) in (0, 1)
            for item in group
        )
        ordered_groups.append(
            {
                "root_seed": root_seed,
                "outcomes": group_outcomes,
                "lengths": [int(item["length"]) for item in group],
            }
        )

    outcome_array = np.asarray(outcomes, dtype=np.float64)
    advantages = leave_one_out_advantages(outcome_array)
    group_sum_residual = float(np.abs(advantages.sum(axis=1)).max())
    advantage_values_finite = bool(np.isfinite(advantages).all())
    nonzero_advantages = int(np.count_nonzero(advantages))
    mixed_groups = int(
        sum(np.unique(group_outcomes).size == 2 for group_outcomes in outcome_array)
    )
    successes = int(outcome_array.sum())
    failures = int(outcome_array.size - successes)
    weight_ess = absolute_weight_ess(advantages)

    pairwise = np.asarray(tuple(first_chunk_pairwise_rms), dtype=np.float64)
    expected_pairs = expected_group_count * expected_group_size * (expected_group_size - 1) // 2
    if pairwise.shape != (expected_pairs,) or not np.isfinite(pairwise).all():
        raise ValueError("first-chunk pairwise RMS values have an invalid manifest")
    median_pairwise_rms = float(np.median(pairwise))

    validity_gates = {
        "episodes_complete_binary": bool(episodes_complete),
        "initial_observations_exact_within_group": bool(initial_hashes_exact),
        "initial_privileged_states_exact_within_group": bool(privileged_hashes_exact),
        "candidate_source_distribution": bool(
            np.isfinite(source_mean)
            and np.isfinite(source_std)
            and abs(source_mean) <= 0.05
            and 0.95 <= source_std <= 1.05
        ),
        "first_chunk_action_diversity": median_pairwise_rms >= 0.05,
        "leave_one_out_zero_sum": advantage_values_finite and group_sum_residual <= 1e-7,
        "policy_parameters_bitwise_unchanged": bool(
            policy_parameters_bitwise_unchanged
        ),
    }
    signal_gates = {
        "success_and_failure_support": successes >= 6 and failures >= 6,
        "mixed_group_count": mixed_groups >= 5,
        "nonzero_advantage_count": nonzero_advantages >= 20,
        "absolute_weight_ess": weight_ess >= 15.0,
    }
    validity_passed = all(validity_gates.values())
    signal_passed = all(signal_gates.values()) if formal else None
    for row, group_advantages in zip(ordered_groups, advantages):
        row["leave_one_out_advantages"] = group_advantages.tolist()

    return {
        "formal": formal,
        "num_groups": expected_group_count,
        "group_size": expected_group_size,
        "num_episodes": len(records),
        "successes": successes,
        "failures": failures,
        "mixed_groups": mixed_groups,
        "nonzero_advantages": nonzero_advantages,
        "absolute_weight_ess": weight_ess,
        "source_mean": float(source_mean),
        "source_std": float(source_std),
        "median_first_chunk_pairwise_normalized_rms": median_pairwise_rms,
        "maximum_group_advantage_sum_residual": group_sum_residual,
        "groups": ordered_groups,
        "validity": {"gates": validity_gates, "passed": validity_passed},
        "signal": {"gates": signal_gates, "passed": signal_passed},
        "passed": validity_passed and (bool(signal_passed) if formal else True),
    }

