#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def _pairable_preupdate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in record.items()
            if key != "weighted_cfm_to_gae_gradient_norm_ratio"
        }
        for record in records
    ]


def analyze(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if control["reflow_regularization_coefficient"] != 0.0:
        raise ValueError("control Reflow coefficient must be zero")
    if candidate["reflow_regularization_coefficient"] != 0.04:
        raise ValueError("candidate Reflow coefficient must be 0.04")
    if control["objective_extra_actor_forwards"] != 0 or candidate[
        "objective_extra_actor_forwards"
    ] != 0:
        raise ValueError("ReFPO objective must not add actor forwards")
    if control["pairing_fingerprint"] != candidate["pairing_fingerprint"]:
        raise ValueError("control and candidate pairing fingerprints differ")
    if _pairable_preupdate(control["preupdate_batches"]) != _pairable_preupdate(
        candidate["preupdate_batches"]
    ):
        raise ValueError("control and candidate pre-update records differ")
    if len(control["epochs"]) != 10 or len(candidate["epochs"]) != 10:
        raise ValueError("H61 requires exactly ten actor epochs")
    if not _finite(control) or not _finite(candidate):
        raise ValueError("audit contains a non-finite value")

    control_final = control["epochs"][-1]
    candidate_final = candidate["epochs"][-1]
    if control_final["epoch"] != 10 or candidate_final["epoch"] != 10:
        raise ValueError("final audit record must be epoch 10")
    batch_results = []
    for control_batch, candidate_batch in zip(
        control_final["batches"], candidate_final["batches"], strict=True
    ):
        if control_batch["batch"] != candidate_batch["batch"]:
            raise ValueError("control and candidate batch identifiers differ")
        control_gain = control_batch["surrogate_gain"]
        candidate_gain = candidate_batch["surrogate_gain"]
        gates = {
            "median_absolute_log_ratio": (
                candidate_batch["median_absolute_log_ratio"]
                <= 0.8 * control_batch["median_absolute_log_ratio"]
            ),
            "ratio_std": (
                candidate_batch["ratio_std"] <= 0.8 * control_batch["ratio_std"]
            ),
            "clip_fraction": (
                candidate_batch["clip_fraction"]
                <= control_batch["clip_fraction"] - 0.10
            ),
            "surrogate_retention": (
                control_gain > 0
                and candidate_gain > 0
                and candidate_gain >= 0.5 * control_gain
            ),
            "outcome_gradient_non_degradation": (
                candidate_batch["outcome_gradient_cosine_to_preupdate"]
                >= control_batch["outcome_gradient_cosine_to_preupdate"] - 0.02
            ),
        }
        batch_results.append(
            {
                "batch": control_batch["batch"],
                "control": control_batch,
                "candidate": candidate_batch,
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    return {
        "hypothesis": "H61",
        "pairing_fingerprint": control["pairing_fingerprint"],
        "exact_preupdate_pairing": True,
        "objective_extra_actor_forwards": 0,
        "control_coefficient": 0.0,
        "candidate_coefficient": 0.04,
        "batches": batch_results,
        "passed": all(batch["passed"] for batch in batch_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(_load(args.control), _load(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
