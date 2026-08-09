from copy import deepcopy

import pytest

from research_scripts.analyze_refpo_audit import analyze


def _record(coefficient: float, *, candidate: bool) -> dict:
    pre = {
        "batch": 0,
        "ratio_mean": 1.0,
        "ratio_std": 0.0,
        "surrogate": 0.0,
        "gae_gradient_norm": 2.0,
        "outcome_gradient_norm": 1.0,
        "unweighted_cfm_loss": 0.2,
        "unweighted_cfm_gradient_norm": 0.5,
        "gae_cfm_gradient_cosine": -0.2,
        "weighted_cfm_to_gae_gradient_norm_ratio": coefficient * 0.25,
    }
    final = {
        "batch": 0,
        "ratio_mean": 1.01,
        "ratio_std": 0.04 if candidate else 0.1,
        "median_absolute_log_ratio": 0.03 if candidate else 0.05,
        "clip_fraction": 0.2 if candidate else 0.4,
        "surrogate": 0.06 if candidate else 0.1,
        "surrogate_gain": 0.06 if candidate else 0.1,
        "gae_gradient_norm": 1.0,
        "outcome_gradient_norm": 0.8,
        "outcome_gradient_cosine_to_preupdate": 0.7,
        "unweighted_cfm_loss": 0.1,
        "unweighted_cfm_gradient_norm": 0.4,
    }
    return {
        "reflow_regularization_coefficient": coefficient,
        "objective_extra_actor_forwards": 0,
        "pairing_fingerprint": "paired",
        "preupdate_batches": [pre, {**pre, "batch": 1}],
        "epochs": [
            {"epoch": epoch, "batches": [final, {**final, "batch": 1}]}
            for epoch in range(1, 11)
        ],
    }


def test_analyze_accepts_a_fully_passing_pair() -> None:
    result = analyze(_record(0.0, candidate=False), _record(0.04, candidate=True))
    assert result["passed"]
    assert all(result["batches"][0]["gates"].values())


def test_analyze_rejects_pairing_mismatch() -> None:
    control = _record(0.0, candidate=False)
    candidate = _record(0.04, candidate=True)
    candidate = deepcopy(candidate)
    candidate["pairing_fingerprint"] = "different"
    with pytest.raises(ValueError, match="fingerprints differ"):
        analyze(control, candidate)
