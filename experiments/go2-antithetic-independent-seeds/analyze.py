"""Analyze the preregistered H69 independent-training-seed confirmation."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


SEEDS = {
    43: {"seed": 20261243, "source_seed": 20261343, "secondary_source_seed": 20261443},
    44: {"seed": 20261244, "source_seed": 20261344, "secondary_source_seed": 20261444},
    45: {"seed": 20261245, "source_seed": 20261345, "secondary_source_seed": 20261445},
}
MODES = ("zero", "random", "iid_pair", "antithetic")
EXPECTED_EPISODES = 512
BOOTSTRAP_SEED = 20261500
BOOTSTRAP_SAMPLES = 20_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--h67-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--archival",
        action="store_true",
        help="reanalyze archived records without reading remote checkpoint files",
    )
    return parser.parse_args()


def _load_manifest(results_dir: Path, training_seed: int, *, archival: bool) -> dict:
    path = results_dir / f"training_seed{training_seed}.json"
    manifest = json.loads(path.read_text())
    expected = {
        "hypothesis": "H69",
        "training_seed": training_seed,
        "num_envs": 4096,
        "iterations": 1500,
        "post_training_checkpoint_sweep": False,
        "passed": True,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}")
    checkpoint = Path(manifest.get("final_checkpoint", ""))
    digest = manifest.get("final_checkpoint_sha256")
    checkpoint_bytes = manifest.get("final_checkpoint_bytes")
    if checkpoint.name != "model_1499.pt":
        raise ValueError(f"{path}: invalid final checkpoint")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{path}: invalid checkpoint digest")
    if type(checkpoint_bytes) is not int or checkpoint_bytes <= 0:
        raise ValueError(f"{path}: invalid checkpoint byte size")
    if archival:
        return manifest
    if not checkpoint.is_file():
        raise ValueError(f"{path}: invalid final checkpoint")
    if checkpoint.stat().st_size != checkpoint_bytes:
        raise ValueError(f"{path}: checkpoint byte size differs")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{path}: checkpoint digest differs")
    return manifest


def _load_evaluation(
    path: Path, expected_mode: str, training_seed: int, checkpoint: str
) -> tuple[dict, np.ndarray]:
    artifact = json.loads(path.read_text())
    expected = {
        **SEEDS[training_seed],
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
    if Path(artifact["checkpoint"]).resolve() != Path(checkpoint).resolve():
        raise ValueError(f"{path}: checkpoint differs from training manifest")
    if set(artifact.get("modes", {})) != {expected_mode}:
        raise ValueError(f"{path}: expected only {expected_mode}")
    mode = artifact["modes"][expected_mode]
    endpoints = 2 if expected_mode in ("iid_pair", "antithetic") else 1
    if mode.get("endpoint_count_per_action") != endpoints:
        raise ValueError(f"{path}: wrong endpoint count")
    if mode.get("nfe_per_action") != 64 * endpoints:
        raise ValueError(f"{path}: wrong NFE accounting")
    if (
        mode.get("episodes") != EXPECTED_EPISODES
        or mode.get("actions_finite") is not True
    ):
        raise ValueError(f"{path}: incomplete episodes or non-finite actions")
    returns = np.asarray(mode.get("episode_returns"), dtype=np.float64)
    if returns.shape != (EXPECTED_EPISODES,) or not np.isfinite(returns).all():
        raise ValueError(f"{path}: invalid returns")
    return artifact, returns


def _paired_summary(difference: np.ndarray, rng: np.random.Generator) -> dict:
    indices = rng.integers(
        0, difference.size, size=(BOOTSTRAP_SAMPLES, difference.size)
    )
    samples = difference[indices].mean(axis=1)
    return {
        "gain": float(difference.mean()),
        "sem": float(difference.std(ddof=1) / np.sqrt(difference.size)),
        "bootstrap_95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def _stratified_summary(
    differences: dict[int, np.ndarray], rng: np.random.Generator
) -> dict:
    seed_samples = []
    for difference in differences.values():
        indices = rng.integers(
            0, difference.size, size=(BOOTSTRAP_SAMPLES, difference.size)
        )
        seed_samples.append(difference[indices].mean(axis=1))
    bootstrap = np.mean(seed_samples, axis=0)
    seed_gains = {str(seed): float(value.mean()) for seed, value in differences.items()}
    return {
        "gain": float(np.mean(list(seed_gains.values()))),
        "seed_gains": seed_gains,
        "bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def analyze(results_dir: Path, h67_analysis_path: Path, *, archival: bool = False) -> dict:
    manifests = {
        seed: _load_manifest(results_dir, seed, archival=archival) for seed in SEEDS
    }
    checkpoint_hashes = {
        manifest["final_checkpoint_sha256"] for manifest in manifests.values()
    }
    if len(checkpoint_hashes) != len(SEEDS):
        raise ValueError("independent training seeds produced duplicate checkpoints")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_results = {}
    differences = {
        "antithetic_minus_iid_pair": {},
        "antithetic_minus_random": {},
        "antithetic_minus_zero": {},
    }
    validity = {}
    for training_seed, manifest in manifests.items():
        artifacts = {}
        returns = {}
        for mode in MODES:
            path = results_dir / f"seed_{training_seed}" / f"{mode}.json"
            artifacts[mode], returns[mode] = _load_evaluation(
                path, mode, training_seed, manifest["final_checkpoint"]
            )
        mode_records = {mode: artifacts[mode]["modes"][mode] for mode in MODES}
        initial_hashes = {
            mode: record["initial_observation_sha256"]
            for mode, record in mode_records.items()
        }
        primary_hashes = {
            mode: mode_records[mode]["source_stream_sha256"]
            for mode in ("random", "iid_pair", "antithetic")
        }
        if len(set(initial_hashes.values())) != 1:
            raise ValueError(f"seed {training_seed}: initial hashes differ")
        if len(set(primary_hashes.values())) != 1:
            raise ValueError(f"seed {training_seed}: primary source hashes differ")
        comparisons = {
            "antithetic_minus_iid_pair": returns["antithetic"] - returns["iid_pair"],
            "antithetic_minus_random": returns["antithetic"] - returns["random"],
            "antithetic_minus_zero": returns["antithetic"] - returns["zero"],
        }
        for name, difference in comparisons.items():
            differences[name][training_seed] = difference
        seed_results[str(training_seed)] = {
            "mean_returns": {
                mode: float(value.mean()) for mode, value in returns.items()
            },
            "comparisons": {
                name: _paired_summary(difference, rng)
                for name, difference in comparisons.items()
            },
        }
        validity[str(training_seed)] = {
            "initial_observation_hashes_match": True,
            "primary_source_stream_hashes_match": True,
            "equal_pair_nfe": (
                mode_records["iid_pair"]["nfe_per_action"]
                == mode_records["antithetic"]["nfe_per_action"]
                == 128
            ),
        }

    pooled = {
        name: _stratified_summary(values, rng) for name, values in differences.items()
    }
    equal_compute = pooled["antithetic_minus_iid_pair"]
    stochastic = pooled["antithetic_minus_random"]
    deployment = pooled["antithetic_minus_zero"]
    primary_gates = {
        "all_training_and_pairing_validity": all(
            all(checks.values()) for checks in validity.values()
        ),
        "all_antithetic_minus_iid_seed_gains_positive": all(
            gain > 0.0 for gain in equal_compute["seed_gains"].values()
        ),
        "pooled_antithetic_minus_iid_lower_positive": equal_compute["bootstrap_95"][0]
        > 0.0,
        "pooled_antithetic_minus_random_lower_positive": stochastic["bootstrap_95"][0]
        > 0.0,
    }
    deployment_gates = {
        "pooled_antithetic_minus_zero_lower_positive": deployment["bootstrap_95"][0]
        > 0.0
    }
    h67 = json.loads(h67_analysis_path.read_text())
    if h67.get("hypothesis") != "H67" or h67.get("passed") is not True:
        raise ValueError("invalid archived H67 artifact")
    return {
        "hypothesis": "H69",
        "training_seeds": list(SEEDS),
        "episodes_per_method_seed": EXPECTED_EPISODES,
        "total_completed_episodes": len(SEEDS) * len(MODES) * EXPECTED_EPISODES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed_results": seed_results,
        "pooled_stratified": pooled,
        "archived_seed_42": h67["comparisons"],
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
    result = analyze(args.results_dir, args.h67_analysis, archival=args.archival)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
