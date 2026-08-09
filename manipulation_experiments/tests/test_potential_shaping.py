import pytest
import numpy as np

from src.potential_shaping import (
    audit_potential_records,
    collection_fingerprint,
    potential_shaping_term,
)


def _record(seed, step, before, after, sparse, terminal, success=False):
    term = potential_shaping_term(
        before, after, discount=0.9, terminal=terminal
    )
    return {
        "seed": seed,
        "step": step,
        "potential_before": before,
        "potential_next": 0.0 if terminal else after,
        "sparse_reward": sparse,
        "terminal": terminal,
        "success": success,
        "shaping_term": term,
        "shaped_reward": sparse + term,
    }


def test_potential_term_zeros_terminal_potential():
    assert potential_shaping_term(0.4, 0.7, discount=0.9, terminal=True) == -0.4


def test_audit_checks_telescoping_and_signed_activity():
    records = []
    for seed in range(2):
        records.extend(
            [
                _record(seed, 0, 0.1, 0.2, 0.0, False),
                _record(seed, 1, 0.2, 0.1, 0.0, False),
                _record(seed, 2, 0.1, 0.0, 0.0, True),
            ]
        )
    result = audit_potential_records(records, {0, 1}, discount=0.9)
    assert result["passed"]
    assert result["active_fraction"] == 1.0
    assert result["positive_fraction"] == 0.5
    assert result["negative_fraction"] == 0.5
    assert result["max_abs_telescoping_residual"] == pytest.approx(0.0)


def test_audit_rejects_incomplete_episode():
    record = _record(0, 0, 0.1, 0.2, 0.0, False)
    with pytest.raises(RuntimeError, match="complete episode"):
        audit_potential_records([record], {0}, discount=0.9)


def test_collection_fingerprint_is_stable_and_sensitive():
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    second = np.array([False, True])
    expected = collection_fingerprint(first, second)
    assert collection_fingerprint(first.copy(), second.copy()) == expected
    changed = first.copy()
    changed[0, 0] = 1.0
    assert collection_fingerprint(changed, second) != expected
