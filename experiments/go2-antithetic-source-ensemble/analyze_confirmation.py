"""Apply the frozen official-scale H66 confirmation gates."""

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_EPISODES = 4096
EVALUATION_SEED = 20261103
SOURCE_SEED = 20261104
BOOTSTRAP_SEED = 20261105
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_BATCH_SIZE = 250


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, expected_mode: str) -> tuple[dict, np.ndarray]:
    artifact = json.loads(path.read_text())
    expected = {
        "seed": EVALUATION_SEED,
        "source_seed": SOURCE_SEED,
        "num_envs": EXPECTED_EPISODES,
        "episodes_per_mode": EXPECTED_EPISODES,
        "integration_method": "euler",
        "sampling_steps": 64,
        "nfe": 64,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}, got {artifact.get(key)!r}")
    if set(artifact.get("modes", {})) != {expected_mode}:
        raise ValueError(f"{path}: expected only mode {expected_mode!r}")
    mode = artifact["modes"][expected_mode]
    expected_endpoints = 2 if expected_mode == "antithetic" else 1
    if mode.get("endpoint_count_per_action") != expected_endpoints:
        raise ValueError(f"{path}: wrong endpoint count")
    if mode.get("episodes") != EXPECTED_EPISODES or mode.get("actions_finite") is not True:
        raise ValueError(f"{path}: incomplete episodes or non-finite actions")
    for key in ("initial_observation_sha256", "source_stream_sha256"):
        if not isinstance(mode.get(key), str) or len(mode[key]) != 64:
            raise ValueError(f"{path}: invalid {key}")
    if not isinstance(mode.get("source_value_count"), int) or mode["source_value_count"] <= 0:
        raise ValueError(f"{path}: invalid source value count")
    returns = np.asarray(mode.get("episode_returns"), dtype=np.float64)
    lengths = np.asarray(mode.get("episode_lengths"), dtype=np.float64)
    if returns.shape != (EXPECTED_EPISODES,) or lengths.shape != (EXPECTED_EPISODES,):
        raise ValueError(f"{path}: incomplete return or length arrays")
    if not np.isfinite(returns).all() or not np.isfinite(lengths).all():
        raise ValueError(f"{path}: non-finite returns or lengths")
    return artifact, returns


def paired_bootstrap(difference: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_SAMPLES)
        indices = rng.integers(
            0, len(difference), size=(stop - start, len(difference))
        )
        means[start:stop] = difference[indices].mean(axis=1)
    return means


def analyze(control_path: Path, candidate_path: Path) -> dict:
    control, control_returns = _load(control_path, "random")
    candidate, candidate_returns = _load(candidate_path, "antithetic")
    for key in ("checkpoint", "task"):
        if control[key] != candidate[key]:
            raise ValueError(f"control and candidate {key} differ")
    control_mode = control["modes"]["random"]
    candidate_mode = candidate["modes"]["antithetic"]
    for key in (
        "initial_observation_sha256",
        "source_stream_sha256",
        "source_value_count",
    ):
        if control_mode[key] != candidate_mode[key]:
            raise ValueError(f"control and candidate {key} differ")

    difference = candidate_returns - control_returns
    bootstrap = paired_bootstrap(difference)
    interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    gain = float(difference.mean())
    gates = {
        "episode_action_return_validity": True,
        "initial_observation_hashes_match": True,
        "complete_source_stream_hashes_match": True,
        "paired_mean_gain_at_least_0_20": gain >= 0.20,
        "paired_bootstrap_lower_bound_positive": interval[0] > 0.0,
    }
    return {
        "hypothesis": "H66-confirmation",
        "control": str(control_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "task": control["task"],
        "episodes_per_method": EXPECTED_EPISODES,
        "total_completed_episodes": 2 * EXPECTED_EPISODES,
        "control_nfe_per_action": 64,
        "candidate_nfe_per_action": 128,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "control_mean": float(control_returns.mean()),
        "candidate_mean": float(candidate_returns.mean()),
        "paired_mean_gain": gain,
        "paired_sem": float(difference.std(ddof=1) / np.sqrt(EXPECTED_EPISODES)),
        "paired_bootstrap_95": interval,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(args.control, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
