import pytest
import torch

from src.discounted_success_critic import (
    discounted_returns_to_observed_terminal,
    spearman_rank_correlation,
)


def test_discounted_returns_exclude_trailing_censored_segment():
    rewards = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    terminals = torch.tensor([[False], [True], [False], [False]])

    returns, valid = discounted_returns_to_observed_terminal(rewards, terminals, 0.5)

    assert returns[:, 0].tolist() == pytest.approx([0.5, 1.0, 0.0, 0.0])
    assert valid[:, 0].tolist() == [True, True, False, False]


def test_discounted_returns_reset_at_each_observed_terminal():
    rewards = torch.tensor([[0.0], [0.0], [0.0], [1.0]])
    terminals = torch.tensor([[False], [True], [False], [True]])

    returns, valid = discounted_returns_to_observed_terminal(rewards, terminals, 0.9)

    assert returns[:, 0].tolist() == pytest.approx([0.0, 0.0, 0.9, 1.0])
    assert valid.all()


def test_discounted_returns_validate_shapes():
    with pytest.raises(ValueError, match="matching"):
        discounted_returns_to_observed_terminal(
            torch.zeros(2, 1), torch.zeros(2), 0.99
        )


def test_spearman_rank_correlation_handles_ties():
    x = torch.tensor([1.0, 2.0, 2.0, 4.0])
    y = torch.tensor([10.0, 20.0, 20.0, 40.0])

    assert spearman_rank_correlation(x, y) == pytest.approx(1.0)


def test_spearman_rank_correlation_detects_reverse_order():
    assert spearman_rank_correlation(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0])
    ) == pytest.approx(-1.0)
