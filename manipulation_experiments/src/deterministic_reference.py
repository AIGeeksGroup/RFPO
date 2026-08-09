from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.group_relative_returns import absolute_weight_ess


def _binary_outcome(value: object, *, context: str) -> int:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(f"{context} outcome must be finite and binary")
    return int(numeric)


def validate_gaussian_manifest(
    records: Sequence[dict[str, object]],
    *,
    expected_root_seeds: Sequence[int],
    replicas: int = 4,
) -> dict[int, list[dict[str, object]]]:
    if replicas < 1 or not expected_root_seeds:
        raise ValueError("expected manifest dimensions must be positive")
    seeds = tuple(int(seed) for seed in expected_root_seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("expected root seeds must be unique")
    if len(records) != len(seeds) * replicas:
        raise ValueError("Gaussian records do not match the expected manifest size")

    grouped: dict[int, list[dict[str, object]]] = {}
    seen: set[tuple[int, int]] = set()
    for record in records:
        root_seed = int(record["root_seed"])
        replica = int(record["replica"])
        key = (root_seed, replica)
        if root_seed not in seeds or key in seen:
            raise ValueError(
                "Gaussian records contain an unexpected or duplicate scenario"
            )
        seen.add(key)
        grouped.setdefault(root_seed, []).append(record)

    for root_seed in seeds:
        group = sorted(
            grouped.get(root_seed, []), key=lambda item: int(item["replica"])
        )
        if [int(record["replica"]) for record in group] != list(range(replicas)):
            raise ValueError(f"root {root_seed} has an invalid replica manifest")
        observation_hashes = {
            str(record["initial_observation_hash"]) for record in group
        }
        privileged_hashes = {
            str(record["initial_privileged_state_hash"]) for record in group
        }
        if len(observation_hashes) != 1 or len(privileged_hashes) != 1:
            raise ValueError(f"root {root_seed} does not have exact initial hashes")
        for record in group:
            first_chunk = np.asarray(
                record["first_normalized_action_chunk"], dtype=np.float64
            )
            success = _binary_outcome(
                record["success"], context=f"root {root_seed} Gaussian"
            )
            if not (
                bool(record["completed"])
                and 1 <= int(record["length"]) <= 400
                and success in (0, 1)
                and first_chunk.size > 0
                and np.isfinite(first_chunk).all()
            ):
                raise ValueError(
                    f"root {root_seed} contains an invalid Gaussian episode"
                )
        grouped[root_seed] = group
    return grouped


def summarize_deterministic_reference_audit(
    gaussian_records: Sequence[dict[str, object]],
    reference_records: Sequence[dict[str, object]],
    *,
    expected_root_seeds: Sequence[int],
    archive_manifest_exact: bool,
    zero_sources_bitwise_exact: bool,
    policy_parameters_bitwise_unchanged: bool,
    formal: bool,
) -> dict[str, object]:
    seeds = tuple(int(seed) for seed in expected_root_seeds)
    grouped = validate_gaussian_manifest(
        gaussian_records,
        expected_root_seeds=seeds,
        replicas=4,
    )
    if len(reference_records) != len(seeds):
        raise ValueError("reference records do not match the expected seed count")

    references: dict[int, dict[str, object]] = {}
    for record in reference_records:
        root_seed = int(record["root_seed"])
        if root_seed not in seeds or root_seed in references:
            raise ValueError(
                "reference records contain an unexpected or duplicate seed"
            )
        references[root_seed] = record
    if set(references) != set(seeds):
        raise ValueError("reference records do not cover every expected seed")

    episodes_complete_binary = True
    initial_observations_exact = True
    initial_privileged_states_exact = True
    first_chunks_finite = True
    rows = []
    all_advantages = []
    reference_successes = 0
    for root_seed in seeds:
        group = grouped[root_seed]
        reference = references[root_seed]
        reference_success = _binary_outcome(
            reference["success"], context=f"root {root_seed} reference"
        )
        first_chunk = np.asarray(
            reference["first_normalized_action_chunk"], dtype=np.float64
        )
        episodes_complete_binary &= bool(reference["completed"])
        episodes_complete_binary &= 1 <= int(reference["length"]) <= 400
        episodes_complete_binary &= reference_success in (0, 1)
        first_chunks_finite &= first_chunk.size > 0 and bool(
            np.isfinite(first_chunk).all()
        )
        initial_observations_exact &= all(
            str(record["initial_observation_hash"])
            == str(reference["initial_observation_hash"])
            for record in group
        )
        initial_privileged_states_exact &= all(
            str(record["initial_privileged_state_hash"])
            == str(reference["initial_privileged_state_hash"])
            for record in group
        )
        gaussian_outcomes = np.asarray(
            [
                _binary_outcome(record["success"], context=f"root {root_seed} Gaussian")
                for record in group
            ],
            dtype=np.float64,
        )
        advantages = gaussian_outcomes - reference_success
        all_advantages.extend(advantages.tolist())
        reference_successes += reference_success
        rows.append(
            {
                "root_seed": root_seed,
                "zero_reference_success": reference_success,
                "zero_reference_length": int(reference["length"]),
                "gaussian_outcomes": gaussian_outcomes.astype(int).tolist(),
                "reference_relative_advantages": advantages.tolist(),
            }
        )

    advantage_array = np.asarray(all_advantages, dtype=np.float64)
    advantages_valid = bool(
        np.isfinite(advantage_array).all()
        and np.isin(advantage_array, [-1.0, 0.0, 1.0]).all()
    )
    positive_advantages = int(np.count_nonzero(advantage_array > 0))
    negative_advantages = int(np.count_nonzero(advantage_array < 0))
    nonzero_advantages = int(np.count_nonzero(advantage_array))
    weight_ess = absolute_weight_ess(advantage_array)
    reference_failures = len(seeds) - reference_successes

    validity_gates = {
        "archive_checksum_and_manifest_exact": bool(archive_manifest_exact),
        "reference_episodes_complete_binary": bool(episodes_complete_binary),
        "initial_observations_exact": bool(initial_observations_exact),
        "initial_privileged_states_exact": bool(initial_privileged_states_exact),
        "zero_sources_bitwise_exact": bool(zero_sources_bitwise_exact),
        "first_action_chunks_finite": bool(first_chunks_finite),
        "reference_advantages_valid": advantages_valid,
        "policy_parameters_bitwise_unchanged": bool(
            policy_parameters_bitwise_unchanged
        ),
    }
    signal_gates = {
        "zero_reference_success_support": reference_successes >= 2,
        "zero_reference_failure_support": reference_failures >= 2,
        "positive_advantage_count": positive_advantages >= 4,
        "negative_advantage_count": negative_advantages >= 4,
        "nonzero_advantage_count": nonzero_advantages >= 20,
        "absolute_weight_ess": weight_ess >= 15.0,
    }
    validity_passed = all(validity_gates.values())
    signal_passed = all(signal_gates.values()) if formal else None
    return {
        "formal": formal,
        "num_scenes": len(seeds),
        "num_gaussian_episodes": len(gaussian_records),
        "num_zero_reference_episodes": len(reference_records),
        "zero_reference_successes": reference_successes,
        "zero_reference_failures": reference_failures,
        "positive_advantages": positive_advantages,
        "negative_advantages": negative_advantages,
        "nonzero_advantages": nonzero_advantages,
        "absolute_weight_ess": weight_ess,
        "scenes": rows,
        "validity": {"gates": validity_gates, "passed": validity_passed},
        "signal": {"gates": signal_gates, "passed": signal_passed},
        "passed": validity_passed and (bool(signal_passed) if formal else True),
    }
