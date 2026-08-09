import numpy as np
import pytest
import torch

from src.vine_returns import (
    array_sha256,
    candidate_rms,
    keyed_seed,
    keyed_standard_normal,
    summarize_vine_differences,
    summarize_vine_records,
)


def test_keyed_noise_is_stable_and_key_sensitive():
    first = keyed_standard_normal((64, 7), 11, 22, 33)
    repeated = keyed_standard_normal((64, 7), 11, 22, 33)
    changed = keyed_standard_normal((64, 7), 11, 22, 34)

    assert keyed_seed(11, 22, 33) == keyed_seed(11, 22, 33)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert array_sha256(first) == array_sha256(repeated.numpy())


def test_candidate_rms_requires_paired_nonempty_chunks():
    assert candidate_rms(np.zeros((2, 3)), np.ones((2, 3))) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        candidate_rms(np.zeros((2, 3)), np.zeros((3, 2)))


def test_vine_difference_gates_pass_for_reproducible_rankings():
    values = [1.0, 0.5, 0.5, -0.5, -1.0, 1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5]
    result = summarize_vine_differences(values, values)

    assert result["passed"]
    assert result["pearson_correlation"] == pytest.approx(1.0)
    assert result["common_sign_agreement"] == pytest.approx(1.0)


def test_vine_difference_gates_reject_inverted_replica():
    values = [1.0, 0.5, 0.5, -0.5, -1.0, 1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5]
    result = summarize_vine_differences(values, [-value for value in values])

    assert not result["passed"]
    assert result["pearson_correlation"] == pytest.approx(-1.0)
    assert result["common_sign_agreement"] == pytest.approx(0.0)


def test_record_summary_enforces_complete_candidate_replica_grid():
    records = []
    for state_index in range(12):
        preferred = state_index % 2
        for candidate in (0, 1):
            for replica in range(4):
                records.append(
                    {
                        "root_seed": 100 + state_index,
                        "branch_step": 80,
                        "candidate": candidate,
                        "continuation_replica": replica,
                        "success": int(candidate == preferred),
                    }
                )
    result = summarize_vine_records(records)
    assert result["num_vine_states"] == 12
    assert result["passed"]

    with pytest.raises(ValueError):
        summarize_vine_records(records[:-1])
