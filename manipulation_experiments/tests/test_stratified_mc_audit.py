import pytest
import torch

from src.stratified_mc_audit import (
    advantage_stratified_sample_counts,
    gradient_estimator_metrics,
)


def test_advantage_stratified_counts_preserve_average_budget():
    advantages = torch.tensor([0.1, -4.0, 2.0, -0.2, 1.0, -3.0])

    counts = advantage_stratified_sample_counts(advantages)

    assert counts.tolist() == [4, 12, 12, 4, 4, 12]
    assert counts.float().mean().item() == 8.0


def test_advantage_stratified_counts_reject_odd_batch():
    with pytest.raises(ValueError, match="even-length"):
        advantage_stratified_sample_counts(torch.ones(3))


def test_gradient_estimator_metrics_match_reference():
    reference = torch.tensor([3.0, 4.0])
    gradients = torch.stack([reference, reference])

    metrics = gradient_estimator_metrics(gradients, reference)

    assert metrics["mean_normalized_gradient_mse"] == 0.0
    assert metrics["median_normalized_gradient_mse"] == 0.0
    assert metrics["mean_gradient_cosine_to_reference"] == pytest.approx(1.0)
    assert metrics["repeat_average_gradient_norm"] == pytest.approx(5.0)
