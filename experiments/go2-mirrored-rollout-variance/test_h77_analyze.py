"""Unit tests for H77 paired-return statistics."""

import importlib.util
import json
from pathlib import Path

import numpy as np


def load_analyzer():
    path = Path(__file__).with_name("analyze.py")
    spec = importlib.util.spec_from_file_location("h77_analyze", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_statistics_detect_mirrored_variance_reduction():
    analyzer = load_analyzer()
    zero = np.linspace(-1.0, 1.0, 256)
    positive = zero + np.sin(np.arange(256))
    negative = zero - np.sin(np.arange(256)) + 0.05 * np.cos(np.arange(256))
    iid = zero + np.cos(np.arange(256))

    result = analyzer.paired_statistics(zero, positive, negative, iid)

    assert result["mirrored_deviation_covariance"] < result["iid_deviation_covariance"]
    assert result["variance_ratio"] < 0.01
    json.dumps(result)
