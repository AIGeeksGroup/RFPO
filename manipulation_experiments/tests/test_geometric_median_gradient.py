import pytest
import torch

from src.median_microbatch_gradient import geometric_median


def test_geometric_median_preserves_symmetric_vector_structure():
    gradients = [
        torch.tensor([-1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, -1.0]),
        torch.tensor([0.0, 1.0]),
    ]

    result = geometric_median(gradients)

    assert result.converged
    torch.testing.assert_close(result.value, torch.zeros(2))


def test_geometric_median_reduces_outlier_objective_from_arithmetic_mean():
    gradients = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([100.0, 100.0]),
    ]
    stacked = torch.stack(gradients)
    arithmetic_mean = stacked.mean(dim=0)

    result = geometric_median(gradients)

    assert result.converged
    geometric_objective = torch.linalg.vector_norm(
        stacked - result.value, dim=1
    ).sum()
    mean_objective = torch.linalg.vector_norm(
        stacked - arithmetic_mean, dim=1
    ).sum()
    assert geometric_objective < mean_objective
    assert torch.linalg.vector_norm(result.value) < torch.linalg.vector_norm(arithmetic_mean)


def test_geometric_median_returns_identical_gradient_without_division_by_zero():
    gradient = torch.tensor([2.0, -3.0])

    result = geometric_median([gradient.clone() for _ in range(4)])

    assert result.converged
    assert result.iterations == 0
    torch.testing.assert_close(result.value, gradient)


@pytest.mark.parametrize(
    "gradients, message",
    [
        ([torch.ones(2)], "at least two"),
        ([torch.ones(2), torch.ones(3)], "matching"),
        ([torch.ones(2, 1), torch.ones(2, 1)], "flat"),
        ([torch.ones(2), torch.tensor([float("nan"), 1.0])], "finite"),
    ],
)
def test_geometric_median_validates_gradients(gradients, message):
    with pytest.raises(ValueError, match=message):
        geometric_median(gradients)

