import numpy as np
import pytest

from src.replanning_phase import analyze_phase_audit


PHASES = (8, 1, 2, 3, 4, 5, 6, 7)


def test_phase_audit_passes_with_locked_control_and_oracle_gates():
    outcomes = np.ones((8, 20), dtype=np.int64)
    outcomes[0, 12:] = 0
    outcomes[1:, 16:] = 0
    hashes = [[f"seed-{seed}" for seed in range(20)] for _ in PHASES]

    analysis = analyze_phase_audit(outcomes, hashes, PHASES)

    assert analysis["control_successes"] == 12
    assert analysis["oracle_successes"] == 16
    assert analysis["oracle_gain"] == 4
    assert analysis["phase_sensitive_seed_count"] == 4
    assert analysis["passed"]


def test_phase_audit_rejects_hash_mismatch_and_constant_outcomes():
    outcomes = np.ones((8, 20), dtype=np.int64)
    hashes = [[f"seed-{seed}" for seed in range(20)] for _ in PHASES]
    hashes[3][7] = "mismatch"

    analysis = analyze_phase_audit(outcomes, hashes, PHASES)

    assert not analysis["initial_observation_hashes_exact"]
    assert analysis["phase_sensitive_seed_count"] == 0
    assert not analysis["passed"]


@pytest.mark.parametrize(
    ("outcomes", "hashes", "message"),
    [
        (np.zeros(8), [["x"]] * 8, "shape"),
        (np.full((8, 2), 2), [["x", "y"]] * 8, "binary"),
        (np.zeros((8, 2)), [["x"]] * 8, "hashes"),
    ],
)
def test_phase_audit_rejects_invalid_inputs(outcomes, hashes, message):
    with pytest.raises(ValueError, match=message):
        analyze_phase_audit(outcomes, hashes, PHASES)
