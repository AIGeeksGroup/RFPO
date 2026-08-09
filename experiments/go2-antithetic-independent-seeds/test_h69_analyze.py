"""Regression tests for strict and archival H69 reanalysis."""

import importlib.util
import json
from pathlib import Path

import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h69_analyze", EXPERIMENT_DIR / "analyze.py")
ANALYZE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZE)


def test_default_mode_rejects_missing_checkpoint():
    with pytest.raises(ValueError, match="invalid final checkpoint"):
        ANALYZE.analyze(
            EXPERIMENT_DIR / "results",
            EXPERIMENT_DIR.parent / "go2-antithetic-attribution/results/analysis.json",
        )


def test_archival_mode_reproduces_committed_analysis_byte_for_byte():
    result = ANALYZE.analyze(
        EXPERIMENT_DIR / "results",
        EXPERIMENT_DIR.parent / "go2-antithetic-attribution/results/analysis.json",
        archival=True,
    )
    rendered = json.dumps(result, indent=2) + "\n"

    assert rendered.encode() == (EXPERIMENT_DIR / "results/analysis.json").read_bytes()
