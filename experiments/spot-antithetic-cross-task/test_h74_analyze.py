"""Regression tests for the H74 analysis artifact."""

import importlib.util
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent


def load_analyzer():
    spec = importlib.util.spec_from_file_location("h74_analyze", EXPERIMENT_DIR / "analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analysis_is_json_serializable():
    result = load_analyzer().analyze(EXPERIMENT_DIR / "results")

    assert type(result["primary_gates"]["healthy_zero64_mean_at_least_250"]) is bool
    json.dumps(result)
