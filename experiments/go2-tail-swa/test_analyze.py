import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("h65_analyze", Path(__file__).with_name("analyze.py"))
ANALYZE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZE)


def _write_artifact(path, checkpoint, mode, returns, *, seed=ANALYZE.EVALUATION_SEED):
    artifact = {
        "checkpoint": checkpoint,
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "seed": seed,
        "source_seed": ANALYZE.SOURCE_SEED,
        "num_envs": ANALYZE.EXPECTED_EPISODES,
        "episodes_per_mode": ANALYZE.EXPECTED_EPISODES,
        "integration_method": "euler",
        "sampling_steps": 64,
        "nfe": 64,
        "modes": {
            mode: {
                "episode_returns": list(map(float, returns)),
                "episode_lengths": [100] * ANALYZE.EXPECTED_EPISODES,
                "actions_finite": True,
                "episodes": ANALYZE.EXPECTED_EPISODES,
                "initial_observation_sha256": "a" * 64,
            }
        },
    }
    path.write_text(json.dumps(artifact))


def _paths(tmp_path, gain):
    paths = {}
    base = np.linspace(10.0, 20.0, ANALYZE.EXPECTED_EPISODES)
    for condition in ("control", "candidate"):
        for mode in ("zero", "random"):
            path = tmp_path / f"{condition}_{mode}.json"
            values = base + (gain if condition == "candidate" else 0.0)
            _write_artifact(path, f"/{condition}.pt", mode, values)
            paths[(condition, mode)] = path
    return paths


def test_positive_paired_gain_passes_all_gates(tmp_path):
    result = ANALYZE.analyze(_paths(tmp_path, 0.2))

    assert result["passed"] is True
    assert result["total_completed_episodes"] == 1024
    assert result["average_mode_gain"] == pytest.approx(0.2)
    assert result["pooled_paired_bootstrap_95"][0] > 0


def test_wrong_seed_is_rejected(tmp_path):
    paths = _paths(tmp_path, 0.2)
    bad_path = paths[("control", "zero")]
    artifact = json.loads(bad_path.read_text())
    artifact["seed"] += 1
    bad_path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="expected seed"):
        ANALYZE.analyze(paths)
