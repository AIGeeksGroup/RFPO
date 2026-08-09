import numpy as np
import pytest

from src.group_relative_returns import (
    absolute_weight_ess,
    leave_one_out_advantages,
    summarize_group_relative_audit,
)


def _records(outcomes):
    records = []
    for group, group_outcomes in enumerate(outcomes):
        for replica, success in enumerate(group_outcomes):
            records.append(
                {
                    "root_seed": 100 + group,
                    "replica": replica,
                    "success": success,
                    "completed": True,
                    "length": 200 + replica,
                    "initial_observation_hash": f"obs-{group}",
                    "initial_privileged_state_hash": f"state-{group}",
                }
            )
    return records


def test_leave_one_out_advantages_have_expected_values_and_zero_sum():
    outcomes = np.asarray([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    advantages = leave_one_out_advantages(outcomes)
    np.testing.assert_allclose(advantages[0], [1.0, -1 / 3, -1 / 3, -1 / 3])
    np.testing.assert_allclose(advantages[1], [2 / 3, 2 / 3, -2 / 3, -2 / 3])
    np.testing.assert_array_equal(advantages[2], np.zeros(4))
    np.testing.assert_allclose(advantages.sum(axis=1), 0.0, atol=1e-12)


def test_leave_one_out_advantages_reject_nonbinary_values():
    with pytest.raises(ValueError, match="finite and binary"):
        leave_one_out_advantages(np.asarray([[0.0, 0.5]]))


def test_absolute_weight_ess_ignores_zero_weights():
    assert absolute_weight_ess(np.asarray([1.0, -1 / 3, -1 / 3, -1 / 3, 0])) == pytest.approx(3.0)
    assert absolute_weight_ess(np.zeros(4)) == 0.0


def test_formal_summary_passes_locked_signal_boundary():
    outcomes = [[1, 0, 0, 0]] * 5 + [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    summary = summarize_group_relative_audit(
        _records(outcomes),
        expected_group_size=4,
        expected_group_count=8,
        source_mean=0.0,
        source_std=1.0,
        first_chunk_pairwise_rms=[0.2] * 48,
        policy_parameters_bitwise_unchanged=True,
        formal=True,
    )
    assert summary["validity"]["passed"]
    assert summary["signal"]["passed"]
    assert summary["mixed_groups"] == 5
    assert summary["nonzero_advantages"] == 20
    assert summary["absolute_weight_ess"] == pytest.approx(15.0)
    assert summary["passed"]


def test_smoke_ignores_formal_signal_gates_but_not_validity():
    summary = summarize_group_relative_audit(
        _records([[0, 0], [0, 0]]),
        expected_group_size=2,
        expected_group_count=2,
        source_mean=0.0,
        source_std=1.0,
        first_chunk_pairwise_rms=[0.2, 0.2],
        policy_parameters_bitwise_unchanged=True,
        formal=False,
    )
    assert summary["validity"]["passed"]
    assert summary["signal"]["passed"] is None
    assert summary["passed"]

