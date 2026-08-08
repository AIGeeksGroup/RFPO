import pytest
import torch

from src.terminal_consistency_filter import terminal_consistent_weights


def test_terminal_consistency_keeps_only_matching_nonzero_signs():
    advantages = torch.tensor([2.0, -3.0, 4.0, -5.0, 6.0])
    returns = torch.tensor([1.0, -1.0, -1.0, 1.0, 0.0])

    weights, retained = terminal_consistent_weights(advantages, returns)

    torch.testing.assert_close(weights, torch.tensor([2.0, -3.0, 0.0, 0.0, 0.0]))
    assert retained.tolist() == [True, True, False, False, False]


def test_terminal_consistency_preserves_dtype_and_device():
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
    weights, retained = terminal_consistent_weights(
        advantages, torch.tensor([2.0, -2.0], dtype=torch.float64)
    )

    assert weights.dtype == advantages.dtype
    assert weights.device == advantages.device
    assert retained.dtype == torch.bool


@pytest.mark.parametrize(
    "advantages, returns, message",
    [
        (torch.ones(2), torch.ones(3), "matching"),
        (torch.ones(2, 1), torch.ones(2, 1), "vectors"),
        (torch.tensor([float('nan')]), torch.ones(1), "finite"),
    ],
)
def test_terminal_consistency_validates_inputs(advantages, returns, message):
    with pytest.raises(ValueError, match=message):
        terminal_consistent_weights(advantages, returns)
