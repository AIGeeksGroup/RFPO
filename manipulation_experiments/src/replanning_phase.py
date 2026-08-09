"""Analysis helpers for the locked replanning phase-shift audit."""

from collections.abc import Sequence

import numpy as np


def analyze_phase_audit(
    outcomes: np.ndarray,
    observation_hashes: Sequence[Sequence[str]],
    phases: Sequence[int],
    *,
    control_phase: int = 8,
) -> dict[str, object]:
    """Evaluate the preregistered H55 validity and phase-sensitivity gates."""
    outcomes = np.asarray(outcomes)
    if outcomes.ndim != 2 or outcomes.shape[0] != len(phases):
        raise ValueError("outcomes must have shape (phase_count, seed_count)")
    if outcomes.shape[1] < 1 or not np.isin(outcomes, [0, 1]).all():
        raise ValueError("outcomes must be a nonempty binary matrix")
    if len(observation_hashes) != len(phases) or any(
        len(row) != outcomes.shape[1] for row in observation_hashes
    ):
        raise ValueError("observation hashes must match the outcome matrix")
    if len(set(phases)) != len(phases) or control_phase not in phases:
        raise ValueError("phases must be unique and include the control phase")

    control_index = list(phases).index(control_phase)
    control_successes = int(outcomes[control_index].sum())
    oracle = outcomes.max(axis=0)
    oracle_successes = int(oracle.sum())
    phase_sensitive = outcomes.min(axis=0) != outcomes.max(axis=0)
    hashes_exact = all(
        len({observation_hashes[row][column] for row in range(len(phases))}) == 1
        for column in range(outcomes.shape[1])
    )
    analysis = {
        "phase_order": list(phases),
        "seed_count": int(outcomes.shape[1]),
        "per_phase_successes": {
            str(phase): int(outcomes[index].sum())
            for index, phase in enumerate(phases)
        },
        "control_phase": control_phase,
        "control_successes": control_successes,
        "oracle_successes": oracle_successes,
        "oracle_gain": oracle_successes - control_successes,
        "phase_sensitive_seed_count": int(phase_sensitive.sum()),
        "phase_sensitive_seed_indices": np.flatnonzero(phase_sensitive).tolist(),
        "initial_observation_hashes_exact": hashes_exact,
    }
    analysis["passed"] = bool(
        hashes_exact
        and control_successes >= 12
        and analysis["phase_sensitive_seed_count"] >= 4
        and analysis["oracle_gain"] >= 3
    )
    return analysis
