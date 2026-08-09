import pytest
import torch

from src.refpo import masked_cfm_mean


def test_masked_cfm_mean_ignores_invalid_action_positions() -> None:
    losses = torch.tensor(
        [
            [[1.0, 3.0], [100.0, 200.0]],
            [[5.0, 7.0], [9.0, 11.0]],
        ]
    )
    valid = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    assert masked_cfm_mean(losses, valid).item() == pytest.approx(6.0)


def test_masked_cfm_mean_preserves_gradients_only_for_valid_entries() -> None:
    losses = torch.arange(8.0).reshape(1, 4, 2).requires_grad_()
    valid = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    masked_cfm_mean(losses, valid).backward()

    assert torch.equal(
        losses.grad,
        torch.tensor([[[0.25, 0.25], [0.0, 0.0], [0.25, 0.25], [0.0, 0.0]]]),
    )


def test_masked_cfm_mean_rejects_empty_or_mismatched_masks() -> None:
    with pytest.raises(ValueError, match="shapes"):
        masked_cfm_mean(torch.ones(2, 3, 4), torch.ones(2, 2))
    with pytest.raises(ValueError, match="at least one"):
        masked_cfm_mean(torch.ones(2, 3, 4), torch.zeros(2, 3))
