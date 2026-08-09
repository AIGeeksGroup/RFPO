"""Analyze the preregistered H67 four-method paired Go2 screen."""

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_EPISODES = 256
EVALUATION_SEED = 20261110
SOURCE_SEED = 20261111
SECONDARY_SOURCE_SEED = 20261112
BOOTSTRAP_SEED = 20261113
BOOTSTRAP_SAMPLES = 20_000
MODES = ("zero", "random", "iid_pair", "antithetic")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in MODES:
        parser.add_argument(f"--{mode.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, expected_mode: str) -> tuple[dict, np.ndarray]:
    artifact = json.loads(path.read_text())
    expected = {
        "seed": EVALUATION_SEED,
        "source_seed": SOURCE_SEED,
        "secondary_source_seed": SECONDARY_SOURCE_SEED,
        "num_envs": EXPECTED_EPISODES,
        "episodes_per_mode": EXPECTED_EPISODES,
        "integration_method": "euler",
        "sampling_steps": 64,
        "nfe": 64,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(
                f"{path}: expected {key}={value!r}, got {artifact.get(key)!r}"
            )
    if set(artifact.get("modes", {})) != {expected_mode}:
        raise ValueError(f"{path}: expected only mode {expected_mode!r}")
    mode = artifact["modes"][expected_mode]
    if (
        mode.get("episodes") != EXPECTED_EPISODES
        or mode.get("actions_finite") is not True
    ):
        raise ValueError(f"{path}: incomplete episodes or non-finite actions")
    expected_endpoints = 2 if expected_mode in ("iid_pair", "antithetic") else 1
    if mode.get("endpoint_count_per_action") != expected_endpoints:
        raise ValueError(f"{path}: wrong endpoint count")
    if mode.get("nfe_per_action") != 64 * expected_endpoints:
        raise ValueError(f"{path}: wrong NFE accounting")
    initial_hash = mode.get("initial_observation_sha256")
    if not isinstance(initial_hash, str) or len(initial_hash) != 64:
        raise ValueError(f"{path}: invalid initial observation hash")
    if expected_mode == "zero":
        if mode.get("source_stream_sha256") is not None:
            raise ValueError(f"{path}: zero mode unexpectedly consumed a source stream")
    else:
        source_hash = mode.get("source_stream_sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError(f"{path}: invalid primary source hash")
    if expected_mode == "iid_pair":
        secondary_hash = mode.get("secondary_source_stream_sha256")
        if not isinstance(secondary_hash, str) or len(secondary_hash) != 64:
            raise ValueError(f"{path}: invalid secondary source hash")
        if mode["secondary_source_value_count"] != mode["source_value_count"]:
            raise ValueError(f"{path}: primary and secondary source counts differ")
        if secondary_hash == mode["source_stream_sha256"]:
            raise ValueError(f"{path}: IID source hashes unexpectedly match")
    elif mode.get("secondary_source_stream_sha256") is not None:
        raise ValueError(f"{path}: unexpected secondary source stream")
    returns = np.asarray(mode.get("episode_returns"), dtype=np.float64)
    if returns.shape != (EXPECTED_EPISODES,) or not np.isfinite(returns).all():
        raise ValueError(f"{path}: incomplete or non-finite returns")
    return artifact, returns


def _paired_summary(candidate, control, bootstrap_indices):
    difference = candidate - control
    bootstrap = difference[bootstrap_indices].mean(axis=1)
    return {
        "gain": float(difference.mean()),
        "sem": float(difference.std(ddof=1) / np.sqrt(EXPECTED_EPISODES)),
        "bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def analyze(paths: dict[str, Path]) -> dict:
    loaded = {mode: _load(paths[mode], mode) for mode in MODES}
    artifacts = {mode: value[0] for mode, value in loaded.items()}
    returns = {mode: value[1] for mode, value in loaded.items()}
    reference = artifacts["zero"]
    for mode, artifact in artifacts.items():
        for key in ("checkpoint", "task"):
            if artifact[key] != reference[key]:
                raise ValueError(f"{mode} and zero {key} differ")
    modes = {mode: artifact["modes"][mode] for mode, artifact in artifacts.items()}
    initial_hashes = {
        mode: value["initial_observation_sha256"] for mode, value in modes.items()
    }
    if len(set(initial_hashes.values())) != 1:
        raise ValueError("initial observation hashes differ")
    primary_hashes = {mode: modes[mode]["source_stream_sha256"] for mode in MODES[1:]}
    if len(set(primary_hashes.values())) != 1:
        raise ValueError("primary source stream hashes differ")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, EXPECTED_EPISODES, size=(BOOTSTRAP_SAMPLES, EXPECTED_EPISODES)
    )
    comparisons = {
        "antithetic_minus_iid_pair": _paired_summary(
            returns["antithetic"], returns["iid_pair"], indices
        ),
        "antithetic_minus_zero": _paired_summary(
            returns["antithetic"], returns["zero"], indices
        ),
        "iid_pair_minus_random": _paired_summary(
            returns["iid_pair"], returns["random"], indices
        ),
        "antithetic_minus_random": _paired_summary(
            returns["antithetic"], returns["random"], indices
        ),
    }
    primary = comparisons["antithetic_minus_iid_pair"]
    gates = {
        "episode_action_return_validity": True,
        "initial_observation_hashes_match": True,
        "primary_source_stream_hashes_match": True,
        "equal_antithetic_iid_nfe": (
            modes["antithetic"]["nfe_per_action"]
            == modes["iid_pair"]["nfe_per_action"]
            == 128
        ),
        "antithetic_minus_iid_point_gain_positive": primary["gain"] > 0.0,
        "antithetic_minus_iid_bootstrap_lower_positive": primary["bootstrap_95"][0]
        > 0.0,
    }
    return {
        "hypothesis": "H67",
        "task": reference["task"],
        "episodes_per_method": EXPECTED_EPISODES,
        "total_completed_episodes": EXPECTED_EPISODES * len(MODES),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "mean_returns": {mode: float(value.mean()) for mode, value in returns.items()},
        "comparisons": comparisons,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = {mode: getattr(args, mode) for mode in MODES}
    result = analyze(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
