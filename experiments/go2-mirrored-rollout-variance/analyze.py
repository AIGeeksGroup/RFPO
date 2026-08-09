"""Analyze the preregistered H77 mirrored closed-loop rollout screen."""

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_ENVIRONMENTS = 256
EVALUATION_SEED = 20261650
SOURCE_SEED = 20261651
SECONDARY_SOURCE_SEED = 20261652
BOOTSTRAP_SEED = 20261653
BOOTSTRAP_SAMPLES = 20_000
MODES = ("zero", "random", "negative_random", "secondary_random")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sample_covariance(left, right):
    return float(np.cov(left, right, ddof=1)[0, 1])


def sample_correlation(left, right):
    return float(np.corrcoef(left, right)[0, 1])


def paired_statistics(zero, positive, negative, iid):
    positive_deviation = positive - zero
    negative_deviation = negative - zero
    iid_deviation = iid - zero
    mirrored_residual = 0.5 * (positive_deviation + negative_deviation)
    iid_residual = 0.5 * (positive_deviation + iid_deviation)
    mirrored_variance = float(np.var(mirrored_residual, ddof=1))
    iid_variance = float(np.var(iid_residual, ddof=1))
    variance_ratio = mirrored_variance / iid_variance

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        positive.size,
        size=(BOOTSTRAP_SAMPLES, positive.size),
    )
    mirrored_bootstrap = np.var(mirrored_residual[indices], axis=1, ddof=1)
    iid_bootstrap = np.var(iid_residual[indices], axis=1, ddof=1)
    ratio_bootstrap = mirrored_bootstrap / iid_bootstrap
    return {
        "mirrored_deviation_covariance": sample_covariance(
            positive_deviation, negative_deviation
        ),
        "iid_deviation_covariance": sample_covariance(
            positive_deviation, iid_deviation
        ),
        "mirrored_deviation_correlation": sample_correlation(
            positive_deviation, negative_deviation
        ),
        "iid_deviation_correlation": sample_correlation(
            positive_deviation, iid_deviation
        ),
        "mirrored_pair_residual_variance": mirrored_variance,
        "iid_pair_residual_variance": iid_variance,
        "variance_ratio": variance_ratio,
        "variance_ratio_bootstrap_95": np.quantile(
            ratio_bootstrap, [0.025, 0.975]
        ).tolist(),
    }


def analyze(path):
    artifact = json.loads(path.read_text())
    expected = {
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "seed": EVALUATION_SEED,
        "source_seed": SOURCE_SEED,
        "secondary_source_seed": SECONDARY_SOURCE_SEED,
        "num_envs": EXPECTED_ENVIRONMENTS,
        "episodes_per_mode": EXPECTED_ENVIRONMENTS,
        "integration_method": "euler",
        "sampling_steps": 32,
        "nfe": 32,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"expected {key}={value!r}")
    if Path(artifact["checkpoint"]).name != "model_1499.pt":
        raise ValueError("expected fixed final checkpoint")
    if set(artifact.get("modes", {})) != set(MODES):
        raise ValueError("wrong evaluation modes")

    records = artifact["modes"]
    returns = {}
    for mode in MODES:
        record = records[mode]
        if (
            record.get("episodes") != EXPECTED_ENVIRONMENTS
            or record.get("nfe_per_action") != 32
            or not record.get("actions_finite")
        ):
            raise ValueError(f"invalid {mode} record")
        values = np.asarray(record.get("episode_returns"), dtype=np.float64)
        if values.shape != (EXPECTED_ENVIRONMENTS,) or not np.isfinite(values).all():
            raise ValueError(f"invalid {mode} returns")
        returns[mode] = values

    if len({records[mode]["initial_observation_sha256"] for mode in MODES}) != 1:
        raise ValueError("initial observation hashes differ")
    stochastic = ("random", "negative_random", "secondary_random")
    if len({records[mode]["source_stream_sha256"] for mode in stochastic}) != 1:
        raise ValueError("primary base-source hashes differ")
    if records["negative_random"].get("source_negation_exact") is not True:
        raise ValueError("negative source is not an exact sign reversal")
    if (
        records["secondary_random"].get("actual_source_stream_sha256")
        != records["secondary_random"].get("secondary_source_stream_sha256")
    ):
        raise ValueError("secondary actual-source hash differs")

    stats = paired_statistics(
        returns["zero"],
        returns["random"],
        returns["negative_random"],
        returns["secondary_random"],
    )
    gates = {
        "all_validity_checks": True,
        "mirrored_covariance_lower_than_iid": (
            stats["mirrored_deviation_covariance"]
            < stats["iid_deviation_covariance"]
        ),
        "variance_ratio_at_most_0_80": stats["variance_ratio"] <= 0.80,
        "variance_ratio_bootstrap_upper_below_1": (
            stats["variance_ratio_bootstrap_95"][1] < 1.0
        ),
    }
    return {
        "hypothesis": "H77",
        "training_seed": 42,
        "environments_per_mode": EXPECTED_ENVIRONMENTS,
        "total_episodes": EXPECTED_ENVIRONMENTS * len(MODES),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "mean_returns": {
            mode: float(values.mean()) for mode, values in returns.items()
        },
        "statistics": stats,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
