"""Analyze the preregistered H68 cross-checkpoint Go2 screen."""

import argparse
import json
from pathlib import Path

import numpy as np


CHECKPOINTS = {
    500: {"seed": 20261120, "source_seed": 20261130, "secondary_source_seed": 20261140},
    1000: {
        "seed": 20261121,
        "source_seed": 20261131,
        "secondary_source_seed": 20261141,
    },
}
MODES = ("zero", "iid_pair", "antithetic")
EXPECTED_EPISODES = 128
BOOTSTRAP_SEED = 20261150
BOOTSTRAP_SAMPLES = 20_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--h67-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, checkpoint: int, expected_mode: str) -> tuple[dict, np.ndarray]:
    artifact = json.loads(path.read_text())
    expected = {
        **CHECKPOINTS[checkpoint],
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
    if Path(artifact["checkpoint"]).name != f"model_{checkpoint}.pt":
        raise ValueError(f"{path}: wrong checkpoint")
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


def _bootstrap_summary(difference, indices):
    samples = difference[indices].mean(axis=1)
    return {
        "gain": float(difference.mean()),
        "sem": float(difference.std(ddof=1) / np.sqrt(difference.size)),
        "bootstrap_95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def analyze(results_dir: Path, h67_analysis_path: Path) -> dict:
    checkpoint_results = {}
    all_antithetic_iid_differences = []
    validity = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for checkpoint in CHECKPOINTS:
        artifacts = {}
        returns = {}
        for mode in MODES:
            path = results_dir / f"checkpoint_{checkpoint}" / f"{mode}.json"
            artifacts[mode], returns[mode] = _load(path, checkpoint, mode)
        mode_records = {mode: artifacts[mode]["modes"][mode] for mode in MODES}
        initial_hashes = {
            mode: record["initial_observation_sha256"]
            for mode, record in mode_records.items()
        }
        primary_hashes = {
            mode: mode_records[mode]["source_stream_sha256"]
            for mode in ("iid_pair", "antithetic")
        }
        initial_match = len(set(initial_hashes.values())) == 1
        primary_match = len(set(primary_hashes.values())) == 1
        if not initial_match or not primary_match:
            raise ValueError(f"checkpoint {checkpoint}: pairing hashes differ")
        difference = returns["antithetic"] - returns["iid_pair"]
        zero_difference = returns["antithetic"] - returns["zero"]
        indices = rng.integers(
            0, EXPECTED_EPISODES, size=(BOOTSTRAP_SAMPLES, EXPECTED_EPISODES)
        )
        checkpoint_results[str(checkpoint)] = {
            "mean_returns": {
                mode: float(value.mean()) for mode, value in returns.items()
            },
            "antithetic_minus_iid_pair": _bootstrap_summary(difference, indices),
            "antithetic_minus_zero": _bootstrap_summary(zero_difference, indices),
        }
        all_antithetic_iid_differences.append(difference)
        validity[str(checkpoint)] = {
            "initial_observation_hashes_match": initial_match,
            "primary_source_stream_hashes_match": primary_match,
            "equal_pair_nfe": (
                mode_records["iid_pair"]["nfe_per_action"]
                == mode_records["antithetic"]["nfe_per_action"]
                == 128
            ),
        }

    pooled_difference = np.concatenate(all_antithetic_iid_differences)
    pooled_indices = rng.integers(
        0, pooled_difference.size, size=(BOOTSTRAP_SAMPLES, pooled_difference.size)
    )
    pooled = _bootstrap_summary(pooled_difference, pooled_indices)
    h67 = json.loads(h67_analysis_path.read_text())
    if h67.get("hypothesis") != "H67" or h67.get("passed") is not True:
        raise ValueError("invalid H67 analysis artifact")
    final_comparison = h67["comparisons"]["antithetic_minus_iid_pair"]
    gains = {
        checkpoint: checkpoint_results[str(checkpoint)]["antithetic_minus_iid_pair"][
            "gain"
        ]
        for checkpoint in CHECKPOINTS
    }
    gates = {
        "all_pairing_and_nfe_validity": all(
            all(checks.values()) for checks in validity.values()
        ),
        "checkpoint_500_point_gain_positive": gains[500] > 0.0,
        "checkpoint_1000_point_gain_positive": gains[1000] > 0.0,
        "pooled_bootstrap_lower_positive": pooled["bootstrap_95"][0] > 0.0,
    }
    return {
        "hypothesis": "H68",
        "episodes_per_method_checkpoint": EXPECTED_EPISODES,
        "new_completed_episodes": len(CHECKPOINTS) * len(MODES) * EXPECTED_EPISODES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "checkpoint_results": checkpoint_results,
        "pooled_antithetic_minus_iid_pair": pooled,
        "archived_checkpoint_1499_antithetic_minus_iid_pair": final_comparison,
        "validity": validity,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(args.results_dir, args.h67_analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
