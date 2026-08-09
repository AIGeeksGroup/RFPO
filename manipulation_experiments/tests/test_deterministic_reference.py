import copy

import pytest

from src.deterministic_reference import (
    summarize_deterministic_reference_audit,
    validate_gaussian_manifest,
)


def _records():
    gaussian = []
    references = []
    for group in range(8):
        root_seed = 100 + group
        gaussian_success = int(group % 2 == 1)
        reference_success = int(group in (0, 2, 4))
        for replica in range(4):
            gaussian.append(
                {
                    "root_seed": root_seed,
                    "replica": replica,
                    "success": gaussian_success,
                    "completed": True,
                    "length": 200 + replica,
                    "initial_observation_hash": f"obs-{group}",
                    "initial_privileged_state_hash": f"state-{group}",
                    "first_normalized_action_chunk": [[float(replica)]],
                }
            )
        references.append(
            {
                "root_seed": root_seed,
                "success": reference_success,
                "completed": True,
                "length": 240,
                "initial_observation_hash": f"obs-{group}",
                "initial_privileged_state_hash": f"state-{group}",
                "first_normalized_action_chunk": [[0.0]],
            }
        )
    return gaussian, references


def _summarize(gaussian, references, *, formal=True):
    return summarize_deterministic_reference_audit(
        gaussian,
        references,
        expected_root_seeds=range(100, 108),
        archive_manifest_exact=True,
        zero_sources_bitwise_exact=True,
        policy_parameters_bitwise_unchanged=True,
        formal=formal,
    )


def test_formal_summary_passes_locked_signal_gates():
    gaussian, references = _records()
    result = _summarize(gaussian, references)

    assert result["validity"]["passed"]
    assert result["signal"]["passed"]
    assert result["zero_reference_successes"] == 3
    assert result["positive_advantages"] == 16
    assert result["negative_advantages"] == 12
    assert result["nonzero_advantages"] == 28
    assert result["absolute_weight_ess"] == pytest.approx(28.0)
    assert result["passed"]


def test_smoke_ignores_signal_failure_but_preserves_validity():
    gaussian, references = _records()
    for record in gaussian:
        record["success"] = 0
    for record in references:
        record["success"] = 0

    result = _summarize(gaussian, references, formal=False)

    assert result["validity"]["passed"]
    assert result["signal"]["passed"] is None
    assert result["passed"]


def test_initial_hash_mismatch_fails_validity():
    gaussian, references = _records()
    references[0]["initial_observation_hash"] = "different"

    result = _summarize(gaussian, references)

    assert not result["validity"]["gates"]["initial_observations_exact"]
    assert not result["passed"]


def test_gaussian_manifest_rejects_duplicate_scenario():
    gaussian, _ = _records()
    invalid = copy.deepcopy(gaussian)
    invalid[-1] = copy.deepcopy(invalid[0])

    with pytest.raises(ValueError, match="unexpected or duplicate"):
        validate_gaussian_manifest(
            invalid,
            expected_root_seeds=range(100, 108),
            replicas=4,
        )


def test_manifest_rejects_nonbinary_outcome_without_integer_coercion():
    gaussian, _ = _records()
    gaussian[0]["success"] = 0.5

    with pytest.raises(ValueError, match="finite and binary"):
        validate_gaussian_manifest(
            gaussian,
            expected_root_seeds=range(100, 108),
            replicas=4,
        )
