import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).with_name("analyze_confirmation.py")
SPEC = importlib.util.spec_from_file_location("h66_confirmation", SCRIPT)
ANALYZE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZE)


def _write(path, mode, returns):
    path.write_text(
        json.dumps(
            {
                "checkpoint": "/model_1499.pt",
                "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
                "seed": ANALYZE.EVALUATION_SEED,
                "source_seed": ANALYZE.SOURCE_SEED,
                "num_envs": ANALYZE.EXPECTED_EPISODES,
                "episodes_per_mode": ANALYZE.EXPECTED_EPISODES,
                "integration_method": "euler",
                "sampling_steps": 64,
                "nfe": 64,
                "modes": {
                    mode: {
                        "episodes": ANALYZE.EXPECTED_EPISODES,
                        "actions_finite": True,
                        "episode_returns": list(map(float, returns)),
                        "episode_lengths": [100] * ANALYZE.EXPECTED_EPISODES,
                        "initial_observation_sha256": "a" * 64,
                        "source_stream_sha256": "b" * 64,
                        "source_value_count": 123456,
                        "endpoint_count_per_action": 2 if mode == "antithetic" else 1,
                    }
                },
            }
        )
    )


def test_official_scale_gain_passes(tmp_path):
    control = tmp_path / "control.json"
    candidate = tmp_path / "candidate.json"
    values = np.linspace(30.0, 50.0, ANALYZE.EXPECTED_EPISODES)
    _write(control, "random", values)
    _write(candidate, "antithetic", values + 0.4)

    result = ANALYZE.analyze(control, candidate)

    assert result["passed"] is True
    assert result["paired_mean_gain"] == pytest.approx(0.4)
    assert result["total_completed_episodes"] == 8192
    assert result["paired_bootstrap_95"][0] > 0
