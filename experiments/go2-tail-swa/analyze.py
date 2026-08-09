"""Apply the preregistered paired H65 reward-screen gates."""

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_EPISODES = 256
EVALUATION_SEED = 20261090
SOURCE_SEED = 20261091
BOOTSTRAP_SEED = 20261092
BOOTSTRAP_SAMPLES = 20_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for condition in ("control", "candidate"):
        for mode in ("zero", "random"):
            parser.add_argument(f"--{condition}-{mode}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_mode(path: Path, expected_mode: str) -> tuple[dict, np.ndarray]:
    artifact = json.loads(path.read_text())
    expected_metadata = {
        "seed": EVALUATION_SEED,
        "source_seed": SOURCE_SEED,
        "num_envs": EXPECTED_EPISODES,
        "episodes_per_mode": EXPECTED_EPISODES,
        "integration_method": "euler",
        "sampling_steps": 64,
        "nfe": 64,
    }
    for key, expected in expected_metadata.items():
        if artifact.get(key) != expected:
            raise ValueError(f"{path}: expected {key}={expected!r}, got {artifact.get(key)!r}")
    if set(artifact.get("modes", {})) != {expected_mode}:
        raise ValueError(f"{path}: expected only mode {expected_mode!r}")
    mode = artifact["modes"][expected_mode]
    if mode.get("episodes") != EXPECTED_EPISODES:
        raise ValueError(f"{path}: incomplete episode count")
    if mode.get("actions_finite") is not True:
        raise ValueError(f"{path}: actions are not all finite")
    initial_hash = mode.get("initial_observation_sha256")
    if not isinstance(initial_hash, str) or len(initial_hash) != 64:
        raise ValueError(f"{path}: missing initial observation SHA-256")
    returns = np.asarray(mode.get("episode_returns"), dtype=np.float64)
    lengths = np.asarray(mode.get("episode_lengths"), dtype=np.float64)
    if returns.shape != (EXPECTED_EPISODES,) or lengths.shape != (EXPECTED_EPISODES,):
        raise ValueError(f"{path}: expected {EXPECTED_EPISODES} returns and lengths")
    if not np.isfinite(returns).all() or not np.isfinite(lengths).all():
        raise ValueError(f"{path}: non-finite returns or lengths")
    return artifact, returns


def analyze(paths: dict[tuple[str, str], Path]) -> dict:
    artifacts: dict[tuple[str, str], dict] = {}
    returns: dict[tuple[str, str], np.ndarray] = {}
    for key, path in paths.items():
        artifacts[key], returns[key] = _load_mode(path, key[1])

    tasks = {artifact["task"] for artifact in artifacts.values()}
    if len(tasks) != 1:
        raise ValueError(f"evaluation tasks differ: {sorted(tasks)}")
    for mode in ("zero", "random"):
        if artifacts[("control", mode)]["checkpoint"] == artifacts[("candidate", mode)]["checkpoint"]:
            raise ValueError(f"{mode}: control and candidate checkpoint paths are identical")
        control_hash = artifacts[("control", mode)]["modes"][mode]["initial_observation_sha256"]
        candidate_hash = artifacts[("candidate", mode)]["modes"][mode]["initial_observation_sha256"]
        if control_hash != candidate_hash:
            raise ValueError(f"{mode}: initial observation hashes differ")
        for condition in ("control", "candidate"):
            if artifacts[(condition, mode)]["checkpoint"] != artifacts[(condition, "zero")]["checkpoint"]:
                raise ValueError(f"{condition}: checkpoint path differs across modes")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mode_results = {}
    differences = []
    for mode in ("zero", "random"):
        control = returns[("control", mode)]
        candidate = returns[("candidate", mode)]
        difference = candidate - control
        indices = rng.integers(
            0, EXPECTED_EPISODES, size=(BOOTSTRAP_SAMPLES, EXPECTED_EPISODES)
        )
        bootstrap = difference[indices].mean(axis=1)
        mode_results[mode] = {
            "control_mean": float(control.mean()),
            "candidate_mean": float(candidate.mean()),
            "paired_mean_gain": float(difference.mean()),
            "paired_sem": float(difference.std(ddof=1) / np.sqrt(EXPECTED_EPISODES)),
            "paired_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
        differences.append(difference)

    pooled = np.concatenate(differences)
    indices = rng.integers(0, len(pooled), size=(BOOTSTRAP_SAMPLES, len(pooled)))
    pooled_bootstrap = pooled[indices].mean(axis=1)
    pooled_interval = np.quantile(pooled_bootstrap, [0.025, 0.975]).tolist()
    average_mode_gain = float(
        np.mean([mode_results[mode]["paired_mean_gain"] for mode in ("zero", "random")])
    )
    gates = {
        "all_episode_action_return_validity": True,
        "zero_gain_at_least_minus_0_05": mode_results["zero"]["paired_mean_gain"] >= -0.05,
        "random_gain_at_least_minus_0_05": mode_results["random"]["paired_mean_gain"] >= -0.05,
        "average_mode_gain_at_least_0_10": average_mode_gain >= 0.10,
        "pooled_bootstrap_lower_bound_positive": pooled_interval[0] > 0.0,
    }
    return {
        "hypothesis": "H65",
        "task": tasks.pop(),
        "artifacts": {f"{condition}_{mode}": str(path.resolve()) for (condition, mode), path in paths.items()},
        "episodes_per_method_mode": EXPECTED_EPISODES,
        "total_completed_episodes": 4 * EXPECTED_EPISODES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "modes": mode_results,
        "average_mode_gain": average_mode_gain,
        "pooled_paired_bootstrap_95": pooled_interval,
        "gates": gates,
        "initial_observation_hashes_match": True,
        "passed": all(gates.values()),
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = {
        (condition, mode): getattr(args, f"{condition}_{mode}")
        for condition in ("control", "candidate")
        for mode in ("zero", "random")
    }
    result = analyze(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
