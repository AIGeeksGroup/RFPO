"""Compare paired H60 control and candidate evaluation artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261062)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    return parser.parse_args()


def main():
    args = parse_args()
    control = json.loads(args.control.read_text())
    candidate = json.loads(args.candidate.read_text())
    for key in (
        "checkpoint",
        "task",
        "seed",
        "source_seed",
        "num_envs",
        "episodes_per_mode",
    ):
        if control[key] != candidate[key]:
            raise ValueError(f"unmatched {key}: {control[key]} != {candidate[key]}")
    if control["nfe"] != 64 or candidate["nfe"] != 64:
        raise ValueError("H60 requires exactly 64 NFE in both conditions")
    if (
        control["integration_method"] != "euler"
        or candidate["integration_method"] != "midpoint"
    ):
        raise ValueError("expected Euler control and midpoint candidate")

    modes = sorted(set(control["modes"]) & set(candidate["modes"]))
    if modes != ["random", "zero"]:
        raise ValueError(f"expected zero and random modes, got {modes}")
    rng = np.random.default_rng(args.bootstrap_seed)
    mode_results = {}
    pooled_differences = []
    for mode in modes:
        control_returns = np.asarray(
            control["modes"][mode]["episode_returns"], dtype=np.float64
        )
        candidate_returns = np.asarray(
            candidate["modes"][mode]["episode_returns"], dtype=np.float64
        )
        if control_returns.shape != candidate_returns.shape:
            raise ValueError(f"unmatched {mode} episode arrays")
        difference = candidate_returns - control_returns
        indices = rng.integers(
            0, len(difference), size=(args.bootstrap_samples, len(difference))
        )
        bootstrap = difference[indices].mean(axis=1)
        mode_results[mode] = {
            "control_mean": float(control_returns.mean()),
            "candidate_mean": float(candidate_returns.mean()),
            "paired_mean_difference": float(difference.mean()),
            "paired_sem": float(difference.std(ddof=1) / np.sqrt(len(difference))),
            "paired_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
        pooled_differences.append(difference)

    pooled = np.concatenate(pooled_differences)
    pooled_indices = rng.integers(
        0, len(pooled), size=(args.bootstrap_samples, len(pooled))
    )
    pooled_bootstrap = pooled[pooled_indices].mean(axis=1)
    pooled_interval = np.quantile(pooled_bootstrap, [0.025, 0.975]).tolist()
    average_mode_gain = float(
        np.mean([result["paired_mean_difference"] for result in mode_results.values()])
    )
    gates = {
        "zero_nonnegative": mode_results["zero"]["paired_mean_difference"] >= 0.0,
        "random_nonnegative": mode_results["random"]["paired_mean_difference"] >= 0.0,
        "average_mode_gain_at_least_0_5": average_mode_gain >= 0.5,
        "pooled_lower_bound_above_minus_0_25": pooled_interval[0] > -0.25,
    }
    output = {
        "control": str(args.control.resolve()),
        "candidate": str(args.candidate.resolve()),
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "modes": mode_results,
        "average_mode_gain": average_mode_gain,
        "pooled_paired_bootstrap_95": pooled_interval,
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
