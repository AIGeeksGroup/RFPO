#!/usr/bin/env python3
"""Check that key manuscript numbers agree with committed result JSON."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEXT = " ".join((HERE / "main.tex").read_text(encoding="utf-8").split())


def load(relative: str) -> dict:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def require(fragment: str, source: str) -> None:
    normalized = " ".join(fragment.split())
    if normalized not in TEXT:
        raise AssertionError(f"missing manuscript value from {source}: {fragment}")


policy = load("experiments/go2-antithetic-cross-policy-summary/results/analysis.json")
mechanism = load("experiments/go2-antithetic-affine-mechanism/results/analysis.json")
spot = load("experiments/spot-antithetic-cross-task/results/analysis.json")
throughput = load("experiments/go2-antithetic-batched-throughput/results/benchmark.json")
rollout = load("experiments/go2-mirrored-rollout-variance/results/analysis.json")

random_effect = policy["estimands"]["antithetic32_minus_random64_equal_total_nfe"]
iid_effect = policy["estimands"]["antithetic64_minus_iid_pair64_equal_pair_nfe"]
zero_effect = policy["estimands"]["antithetic32_minus_zero32_deployment_control"]

require(f"${random_effect['mean']:.3f}$ return with SD ${random_effect['sample_sd']:.3f}$", "H76")
require(
    f"$[{random_effect['student_t_95'][0]:.3f},{random_effect['student_t_95'][1]:.3f}]$",
    "H76",
)
require(f"mean is ${iid_effect['mean']:.3f}$ (SD ${iid_effect['sample_sd']:.3f}$)", "H76")
require(
    f"interval $[{iid_effect['student_t_95'][0]:.3f},{iid_effect['student_t_95'][1]:.3f}]$",
    "H76",
)
require(f"mean ${zero_effect['mean']:.3f}$ (SD ${zero_effect['sample_sd']:.3f}$)", "H76")

aggregate = mechanism["aggregate"]
require(f"is ${aggregate['displacement_cosine']['mean']:.3f}$ on average", "H75")
require(f"ratio is ${aggregate['antithetic_residual_ratio']['mean']:.3f}$", "H75")
require(f"IID residual ratio is ${aggregate['iid_residual_ratio']['mean']:.3f}$", "H75")

spot_random = spot["comparisons"]["antithetic32_minus_random64"]
spot_iid = spot["comparisons"]["antithetic32_minus_iid_pair32"]
spot_zero = spot["comparisons"]["antithetic32_minus_zero32"]
require(f"random64 by ${spot_random['gain']:.2f}$", "H74")
require(f"only ${spot_iid['gain']:.2f}$", "H74")
require(f"zero32 by ${spot_zero['gain']:.2f}$", "H74")

timings = throughput["timings"]
require(f"random64 takes ${1000 * timings['random64']['median_seconds_per_call']:.3f}$ ms", "H71")
require(
    f"batched symmetric32 takes ${1000 * timings['antithetic32_batched']['median_seconds_per_call']:.3f}$ ms",
    "H71",
)

variance = rollout["statistics"]
require(f"ratio is ${variance['variance_ratio']:.3f}$", "H77")
require(
    f"$[{variance['variance_ratio_bootstrap_95'][0]:.3f},{variance['variance_ratio_bootstrap_95'][1]:.3f}]$",
    "H77",
)

for artifact in (
    "figures/fig_main_results.pdf",
    "figures/fig_main_results.png",
    "figures/fig_efficiency_boundary.pdf",
    "figures/fig_efficiency_boundary.png",
    "main.pdf",
):
    path = HERE / artifact
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"missing or empty paper artifact: {path}")

print("validated manuscript claims and paper artifacts")
