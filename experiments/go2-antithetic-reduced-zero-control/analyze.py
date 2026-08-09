"""Analyze the preregistered H72 reduced-step zero-source control."""

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_EPISODES = 256
EVALUATION_SEED = 20261600
BOOTSTRAP_SEED = 20261612
BOOTSTRAP_SAMPLES = 20_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h70-dir", type=Path, required=True)
    parser.add_argument("--zero32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_mode(path, mode, sampling_steps, nfe_per_action):
    artifact = json.loads(path.read_text())
    expected = {
        "seed": EVALUATION_SEED,
        "num_envs": EXPECTED_EPISODES,
        "episodes_per_mode": EXPECTED_EPISODES,
        "integration_method": "euler",
        "sampling_steps": sampling_steps,
        "nfe": sampling_steps,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}")
    if set(artifact.get("modes", {})) != {mode}:
        raise ValueError(f"{path}: expected only {mode}")
    record = artifact["modes"][mode]
    if record.get("nfe_per_action") != nfe_per_action:
        raise ValueError(f"{path}: wrong total NFE")
    if record.get("episodes") != EXPECTED_EPISODES or not record.get("actions_finite"):
        raise ValueError(f"{path}: incomplete or non-finite evaluation")
    returns = np.asarray(record.get("episode_returns"), dtype=np.float64)
    if returns.shape != (EXPECTED_EPISODES,) or not np.isfinite(returns).all():
        raise ValueError(f"{path}: invalid returns")
    return artifact, record, returns


def paired_summary(candidate, control, indices):
    difference = candidate - control
    bootstrap = difference[indices].mean(axis=1)
    return {
        "gain": float(difference.mean()),
        "sem": float(difference.std(ddof=1) / np.sqrt(difference.size)),
        "bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def analyze(h70_dir, zero32_path):
    h70_analysis = json.loads((h70_dir / "analysis.json").read_text())
    if h70_analysis.get("hypothesis") != "H70" or not h70_analysis.get("passed"):
        raise ValueError("invalid H70 analysis artifact")
    zero64 = load_mode(h70_dir / "zero64.json", "zero", 64, 64)
    antithetic32 = load_mode(h70_dir / "antithetic32.json", "antithetic", 32, 64)
    zero32 = load_mode(zero32_path, "zero", 32, 32)
    loaded = {"zero64": zero64, "antithetic32": antithetic32, "zero32": zero32}
    reference = zero64[0]
    for label, (artifact, record, _) in loaded.items():
        if artifact.get("checkpoint") != reference.get("checkpoint"):
            raise ValueError(f"{label}: checkpoint differs")
        if artifact.get("task") != reference.get("task"):
            raise ValueError(f"{label}: task differs")
        if record.get("initial_observation_sha256") != zero64[1].get(
            "initial_observation_sha256"
        ):
            raise ValueError(f"{label}: initial observation hash differs")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, EXPECTED_EPISODES, size=(BOOTSTRAP_SAMPLES, EXPECTED_EPISODES)
    )
    comparisons = {
        "antithetic32_minus_zero32": paired_summary(
            antithetic32[2], zero32[2], indices
        ),
        "zero32_minus_zero64": paired_summary(zero32[2], zero64[2], indices),
    }
    primary = comparisons["antithetic32_minus_zero32"]
    gates = {
        "all_pairing_and_return_validity": True,
        "antithetic32_minus_zero32_point_positive": primary["gain"] > 0.0,
        "antithetic32_minus_zero32_lower_positive": primary["bootstrap_95"][0] > 0.0,
    }
    return {
        "hypothesis": "H72",
        "episodes_per_method": EXPECTED_EPISODES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "mean_returns": {label: float(value[2].mean()) for label, value in loaded.items()},
        "comparisons": comparisons,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(args.h70_dir, args.zero32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
