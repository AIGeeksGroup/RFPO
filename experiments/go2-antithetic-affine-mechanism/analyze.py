"""Aggregate the preregistered H75 cross-policy mechanism audit."""

import argparse
import json
from pathlib import Path

import numpy as np


TRAINING_SEEDS = (42, 43, 44, 45)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    records = {}
    for training_seed in TRAINING_SEEDS:
        path = args.results_dir / f"seed{training_seed}.json"
        record = json.loads(path.read_text())
        expected = {
            "hypothesis": "H75",
            "training_seed": training_seed,
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "seed": 20261650,
            "source_seed": 20261651,
            "secondary_source_seed": 20261652,
            "num_envs": 128,
            "rollout_steps": 8,
            "sampling_steps": 32,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"{path}: expected {key}={value!r}")
        records[str(training_seed)] = record

    metric_names = (
        "displacement_cosine",
        "antithetic_residual_ratio",
        "iid_residual_ratio",
        "positive_displacement_rms",
        "opposite_displacement_rms",
        "even_displacement_rms",
        "odd_displacement_rms",
    )
    aggregate = {
        name: {
            "mean": float(np.mean([record[name] for record in records.values()])),
            "min": float(np.min([record[name] for record in records.values()])),
            "max": float(np.max([record[name] for record in records.values()])),
        }
        for name in metric_names
    }
    gates = {
        "all_seed_audits_passed": all(record.get("passed") for record in records.values()),
        "all_validity_gates_passed": all(
            all(record.get("gates", {}).values()) for record in records.values()
        ),
    }
    result = {
        "hypothesis": "H75",
        "training_seeds": list(TRAINING_SEEDS),
        "policies": records,
        "aggregate": aggregate,
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
