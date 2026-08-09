"""Analyze the preregistered H74 official Spot cross-task screen."""

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = {
    "zero64": {"mode": "zero", "steps": 64, "nfe": 64},
    "zero32": {"mode": "zero", "steps": 32, "nfe": 32},
    "random64": {"mode": "random", "steps": 64, "nfe": 64},
    "iid_pair32": {"mode": "iid_pair", "steps": 32, "nfe": 64},
    "antithetic32": {"mode": "antithetic", "steps": 32, "nfe": 64},
}
EXPECTED_EPISODES = 256
EVALUATION_SEED = 20261640
SOURCE_SEED = 20261641
SECONDARY_SOURCE_SEED = 20261642
BOOTSTRAP_SEED = 20261643
BOOTSTRAP_SAMPLES = 20_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_mode(path, label, checkpoint):
    spec = METHODS[label]
    artifact = json.loads(path.read_text())
    expected = {
        "checkpoint": checkpoint,
        "task": "Isaac-Velocity-Flat-Spot-v0",
        "seed": EVALUATION_SEED,
        "source_seed": SOURCE_SEED,
        "secondary_source_seed": SECONDARY_SOURCE_SEED,
        "num_envs": EXPECTED_EPISODES,
        "episodes_per_mode": EXPECTED_EPISODES,
        "integration_method": "euler",
        "sampling_steps": spec["steps"],
        "nfe": spec["steps"],
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}")
    if set(artifact.get("modes", {})) != {spec["mode"]}:
        raise ValueError(f"{path}: wrong mode")
    record = artifact["modes"][spec["mode"]]
    if record.get("nfe_per_action") != spec["nfe"]:
        raise ValueError(f"{path}: wrong total NFE")
    if record.get("episodes") != EXPECTED_EPISODES or not record.get("actions_finite"):
        raise ValueError(f"{path}: incomplete or non-finite evaluation")
    returns = np.asarray(record.get("episode_returns"), dtype=np.float64)
    if returns.shape != (EXPECTED_EPISODES,) or not np.isfinite(returns).all():
        raise ValueError(f"{path}: invalid returns")
    return record, returns


def paired_summary(candidate, control, indices):
    difference = candidate - control
    bootstrap = difference[indices].mean(axis=1)
    return {
        "gain": float(difference.mean()),
        "sem": float(difference.std(ddof=1) / np.sqrt(difference.size)),
        "bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def analyze(results_dir):
    manifest = json.loads((results_dir / "training_seed42.json").read_text())
    expected_manifest = {
        "hypothesis": "H74",
        "training_seed": 42,
        "task": "Isaac-Velocity-Flat-Spot-v0",
        "num_envs": 4096,
        "iterations": 1500,
        "post_training_checkpoint_sweep": False,
        "passed": True,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"training manifest: expected {key}={value!r}")
    checkpoint = manifest["final_checkpoint"]
    loaded = {
        label: load_mode(results_dir / f"{label}.json", label, checkpoint)
        for label in METHODS
    }
    records = {label: value[0] for label, value in loaded.items()}
    returns = {label: value[1] for label, value in loaded.items()}
    initial_hashes = {
        record["initial_observation_sha256"] for record in records.values()
    }
    if len(initial_hashes) != 1:
        raise ValueError("initial observation hashes differ")
    source_hashes = {
        records[label]["source_stream_sha256"]
        for label in ("random64", "iid_pair32", "antithetic32")
    }
    if len(source_hashes) != 1:
        raise ValueError("primary source stream hashes differ")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, EXPECTED_EPISODES, size=(BOOTSTRAP_SAMPLES, EXPECTED_EPISODES)
    )
    comparisons = {
        "antithetic32_minus_random64": paired_summary(
            returns["antithetic32"], returns["random64"], indices
        ),
        "antithetic32_minus_iid_pair32": paired_summary(
            returns["antithetic32"], returns["iid_pair32"], indices
        ),
        "antithetic32_minus_zero32": paired_summary(
            returns["antithetic32"], returns["zero32"], indices
        ),
        "zero32_minus_zero64": paired_summary(
            returns["zero32"], returns["zero64"], indices
        ),
    }
    random_comparison = comparisons["antithetic32_minus_random64"]
    iid_comparison = comparisons["antithetic32_minus_iid_pair32"]
    deployment = comparisons["antithetic32_minus_zero32"]
    primary_gates = {
        "healthy_zero64_mean_at_least_250": returns["zero64"].mean() >= 250.0,
        "all_pairing_and_return_validity": True,
        "equal_random_iid_antithetic_total_nfe": (
            records["random64"]["nfe_per_action"]
            == records["iid_pair32"]["nfe_per_action"]
            == records["antithetic32"]["nfe_per_action"]
            == 64
        ),
        "antithetic32_minus_random64_point_positive": random_comparison["gain"] > 0.0,
        "antithetic32_minus_random64_lower_positive": random_comparison["bootstrap_95"][0]
        > 0.0,
        "antithetic32_minus_iid_pair32_point_positive": iid_comparison["gain"] > 0.0,
        "antithetic32_minus_iid_pair32_lower_positive": iid_comparison["bootstrap_95"][0]
        > 0.0,
    }
    deployment_gates = {
        "antithetic32_minus_zero32_lower_positive": deployment["bootstrap_95"][0]
        > 0.0
    }
    return {
        "hypothesis": "H74",
        "training_seed": 42,
        "episodes_per_method": EXPECTED_EPISODES,
        "total_completed_episodes": EXPECTED_EPISODES * len(METHODS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "mean_returns": {label: float(value.mean()) for label, value in returns.items()},
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
