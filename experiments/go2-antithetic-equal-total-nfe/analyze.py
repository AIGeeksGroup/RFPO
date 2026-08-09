"""Analyze the preregistered H70 equal-total-NFE Go2 screen."""

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = {
    "zero64": {"mode": "zero", "sampling_steps": 64, "nfe_per_action": 64},
    "random64": {"mode": "random", "sampling_steps": 64, "nfe_per_action": 64},
    "antithetic32": {
        "mode": "antithetic",
        "sampling_steps": 32,
        "nfe_per_action": 64,
    },
    "antithetic64": {
        "mode": "antithetic",
        "sampling_steps": 64,
        "nfe_per_action": 128,
    },
}
EXPECTED_EPISODES = 256
EVALUATION_SEED = 20261600
SOURCE_SEED = 20261601
BOOTSTRAP_SEED = 20261602
BOOTSTRAP_SAMPLES = 20_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, label: str) -> tuple[dict, np.ndarray]:
    spec = METHODS[label]
    artifact = json.loads(path.read_text())
    expected = {
        "seed": EVALUATION_SEED,
        "source_seed": SOURCE_SEED,
        "num_envs": EXPECTED_EPISODES,
        "episodes_per_mode": EXPECTED_EPISODES,
        "integration_method": "euler",
        "sampling_steps": spec["sampling_steps"],
        "nfe": spec["sampling_steps"],
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(
                f"{path}: expected {key}={value!r}, got {artifact.get(key)!r}"
            )
    if set(artifact.get("modes", {})) != {spec["mode"]}:
        raise ValueError(f"{path}: wrong mode")
    mode = artifact["modes"][spec["mode"]]
    if mode.get("nfe_per_action") != spec["nfe_per_action"]:
        raise ValueError(f"{path}: wrong total NFE")
    if (
        mode.get("episodes") != EXPECTED_EPISODES
        or mode.get("actions_finite") is not True
    ):
        raise ValueError(f"{path}: incomplete episodes or non-finite actions")
    returns = np.asarray(mode.get("episode_returns"), dtype=np.float64)
    if returns.shape != (EXPECTED_EPISODES,) or not np.isfinite(returns).all():
        raise ValueError(f"{path}: invalid returns")
    return artifact, returns


def _paired_summary(candidate, control, indices):
    difference = candidate - control
    bootstrap = difference[indices].mean(axis=1)
    return {
        "gain": float(difference.mean()),
        "sem": float(difference.std(ddof=1) / np.sqrt(EXPECTED_EPISODES)),
        "bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def analyze(results_dir: Path) -> dict:
    loaded = {label: _load(results_dir / f"{label}.json", label) for label in METHODS}
    artifacts = {label: value[0] for label, value in loaded.items()}
    returns = {label: value[1] for label, value in loaded.items()}
    reference = artifacts["zero64"]
    for label, artifact in artifacts.items():
        for key in ("checkpoint", "task"):
            if artifact[key] != reference[key]:
                raise ValueError(f"{label}: {key} differs")
    mode_records = {
        label: artifacts[label]["modes"][METHODS[label]["mode"]] for label in METHODS
    }
    initial_hashes = {
        label: record["initial_observation_sha256"]
        for label, record in mode_records.items()
    }
    if len(set(initial_hashes.values())) != 1:
        raise ValueError("initial observation hashes differ")
    source_hashes = {
        label: mode_records[label]["source_stream_sha256"]
        for label in ("random64", "antithetic32", "antithetic64")
    }
    if len(set(source_hashes.values())) != 1:
        raise ValueError("primary source stream hashes differ")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, EXPECTED_EPISODES, size=(BOOTSTRAP_SAMPLES, EXPECTED_EPISODES)
    )
    comparisons = {
        "antithetic32_minus_random64": _paired_summary(
            returns["antithetic32"], returns["random64"], indices
        ),
        "antithetic32_minus_zero64": _paired_summary(
            returns["antithetic32"], returns["zero64"], indices
        ),
        "antithetic32_minus_antithetic64": _paired_summary(
            returns["antithetic32"], returns["antithetic64"], indices
        ),
        "antithetic64_minus_random64": _paired_summary(
            returns["antithetic64"], returns["random64"], indices
        ),
    }
    primary = comparisons["antithetic32_minus_random64"]
    deployment = comparisons["antithetic32_minus_zero64"]
    primary_gates = {
        "all_pairing_and_return_validity": True,
        "equal_random64_antithetic32_total_nfe": (
            mode_records["random64"]["nfe_per_action"]
            == mode_records["antithetic32"]["nfe_per_action"]
            == 64
        ),
        "antithetic32_minus_random64_point_positive": primary["gain"] > 0.0,
        "antithetic32_minus_random64_lower_positive": primary["bootstrap_95"][0] > 0.0,
    }
    deployment_gates = {
        "antithetic32_minus_zero64_lower_positive": deployment["bootstrap_95"][0] > 0.0
    }
    return {
        "hypothesis": "H70",
        "episodes_per_method": EXPECTED_EPISODES,
        "total_completed_episodes": EXPECTED_EPISODES * len(METHODS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "mean_returns": {
            label: float(value.mean()) for label, value in returns.items()
        },
        "comparisons": comparisons,
        "primary_gates": primary_gates,
        "deployment_gates": deployment_gates,
        "passed": all(primary_gates.values()),
        "deployment_passed": all(deployment_gates.values()),
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
