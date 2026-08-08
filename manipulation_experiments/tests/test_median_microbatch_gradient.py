import pytest
import torch

from src.median_microbatch_gradient import (
    aggregate_microbatch_gradients,
    middle_pair_mean,
)


def test_middle_pair_mean_uses_both_middle_values():
    values = torch.tensor(
        [[1.0, 100.0], [9.0, 4.0], [3.0, 2.0], [5.0, 8.0]]
    )

    torch.testing.assert_close(
        middle_pair_mean(values), torch.tensor([4.0, 6.0])
    )


def test_gradient_aggregation_rejects_coordinate_outlier():
    gradients = [
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 3.0]),
        torch.tensor([1.0, 5.0]),
        torch.tensor([101.0, 7.0]),
    ]

    control, candidate = aggregate_microbatch_gradients(gradients)

    torch.testing.assert_close(control, torch.tensor([26.0, 4.0]))
    torch.testing.assert_close(candidate, torch.tensor([1.0, 4.0]))


@pytest.mark.parametrize(
    "gradients, message",
    [
        ([torch.ones(2)] * 3, "exactly four"),
        ([torch.ones(2), torch.ones(2), torch.ones(2), torch.ones(3)], "matching"),
        ([torch.ones(2, 1)] * 4, "flat"),
    ],
)
def test_gradient_aggregation_validates_inputs(gradients, message):
    with pytest.raises(ValueError, match=message):
        aggregate_microbatch_gradients(gradients)


def test_middle_pair_mean_rejects_odd_count():
    with pytest.raises(ValueError, match="even-sized"):
        middle_pair_mean(torch.ones(3))
