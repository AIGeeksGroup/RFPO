"""Analyze H73 equal-total-NFE confirmation across independent Go2 policies."""

import argparse
import json
from pathlib import Path

import numpy as np


SEEDS = {
    43: {"seed": 20261243, "source_seed": 20261343, "secondary_source_seed": 20261443},
    44: {"seed": 20261244, "source_seed": 20261344, "secondary_source_seed": 20261444},
    45: {"seed": 20261245, "source_seed": 20261345, "secondary_source_seed": 20261445},
}
EXPECTED_EPISODES = 512
BOOTSTRAP_SEED = 20261620
BOOTSTRAP_SAMPLES = 20_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h69-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_mode(path, mode, training_seed, sampling_steps, nfe_per_action, checkpoint):
    artifact = json.loads(path.read_text())
    expected = {
        **SEEDS[training_seed],
        "num_envs": EXPECTED_EPISODES,
        "episodes_per_mode": EXPECTED_EPISODES,
        "integration_method": "euler",
        "sampling_steps": sampling_steps,
        "nfe": sampling_steps,
        "checkpoint": checkpoint,
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
    return record, returns


def paired_summary(difference, rng):
    indices = rng.integers(
        0, difference.size, size=(BOOTSTRAP_SAMPLES, difference.size)
    )
    bootstrap = difference[indices].mean(axis=1)
    return {
        "gain": float(difference.mean()),
        "sem": float(difference.std(ddof=1) / np.sqrt(difference.size)),
        "bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def stratified_summary(differences, rng):
    samples = []
    for difference in differences.values():
        indices = rng.integers(
            0, difference.size, size=(BOOTSTRAP_SAMPLES, difference.size)
        )
        samples.append(difference[indices].mean(axis=1))
    bootstrap = np.mean(samples, axis=0)
    seed_gains = {str(seed): float(value.mean()) for seed, value in differences.items()}
    return {
        "gain": float(np.mean(list(seed_gains.values()))),
        "seed_gains": seed_gains,
        "bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def analyze(h69_dir, results_dir):
    h69_analysis = json.loads((h69_dir / "analysis.json").read_text())
    if h69_analysis.get("hypothesis") != "H69" or not h69_analysis.get("passed"):
        raise ValueError("invalid H69 analysis artifact")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_results = {}
    differences = {
        "antithetic32_minus_random64": {},
        "antithetic32_minus_zero32": {},
        "zero32_minus_zero64": {},
    }
    validity = {}
    for training_seed in SEEDS:
        manifest = json.loads((h69_dir / f"training_seed{training_seed}.json").read_text())
        checkpoint = manifest["final_checkpoint"]
        archived = h69_dir / f"seed_{training_seed}"
        current = results_dir / f"seed_{training_seed}"
        loaded = {
            "random64": load_mode(
                archived / "random.json", "random", training_seed, 64, 64, checkpoint
            ),
            "zero64": load_mode(
                archived / "zero.json", "zero", training_seed, 64, 64, checkpoint
            ),
            "antithetic32": load_mode(
                current / "antithetic32.json",
                "antithetic",
                training_seed,
                32,
                64,
                checkpoint,
            ),
            "zero32": load_mode(
                current / "zero32.json", "zero", training_seed, 32, 32, checkpoint
            ),
        }
        records = {label: value[0] for label, value in loaded.items()}
        returns = {label: value[1] for label, value in loaded.items()}
        initial_hashes = {
            record["initial_observation_sha256"] for record in records.values()
        }
        if len(initial_hashes) != 1:
            raise ValueError(f"seed {training_seed}: initial hashes differ")
        if (
            records["random64"]["source_stream_sha256"]
            != records["antithetic32"]["source_stream_sha256"]
        ):
            raise ValueError(f"seed {training_seed}: source hashes differ")
        comparisons = {
            "antithetic32_minus_random64": returns["antithetic32"] - returns["random64"],
            "antithetic32_minus_zero32": returns["antithetic32"] - returns["zero32"],
            "zero32_minus_zero64": returns["zero32"] - returns["zero64"],
        }
        for name, difference in comparisons.items():
            differences[name][training_seed] = difference
        seed_results[str(training_seed)] = {
            "mean_returns": {label: float(value.mean()) for label, value in returns.items()},
            "comparisons": {
                name: paired_summary(difference, rng)
                for name, difference in comparisons.items()
            },
        }
        validity[str(training_seed)] = {
            "initial_observation_hashes_match": True,
            "primary_source_stream_hashes_match": True,
            "equal_random_antithetic_total_nfe": (
                records["random64"]["nfe_per_action"]
                == records["antithetic32"]["nfe_per_action"]
                == 64
            ),
        }

    pooled = {
        name: stratified_summary(values, rng) for name, values in differences.items()
    }
    primary = pooled["antithetic32_minus_random64"]
    deployment = pooled["antithetic32_minus_zero32"]
    primary_gates = {
        "all_pairing_and_return_validity": all(
            all(checks.values()) for checks in validity.values()
        ),
        "all_antithetic32_minus_random64_seed_gains_positive": all(
            gain > 0.0 for gain in primary["seed_gains"].values()
        ),
        "pooled_antithetic32_minus_random64_lower_positive": primary["bootstrap_95"][0]
        > 0.0,
    }
    deployment_gates = {
        "pooled_antithetic32_minus_zero32_lower_positive": deployment["bootstrap_95"][0]
        > 0.0
    }
    return {
        "hypothesis": "H73",
        "training_seeds": list(SEEDS),
        "episodes_per_new_method_seed": EXPECTED_EPISODES,
        "new_completed_episodes": len(SEEDS) * 2 * EXPECTED_EPISODES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed_results": seed_results,
        "pooled_stratified": pooled,
        "validity": validity,
        "primary_gates": primary_gates,
        "deployment_gates": deployment_gates,
        "passed": all(primary_gates.values()),
        "deployment_passed": all(deployment_gates.values()),
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(args.h69_dir, args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
