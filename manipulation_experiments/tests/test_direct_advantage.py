import pytest
import torch

from src.direct_advantage import (
    DirectAdvantageHead,
    direct_advantage_residuals,
    discounted_macro_rewards,
)


def test_centered_head_is_zero_when_behavior_matches_center_mean():
    torch.manual_seed(0)
    head = DirectAdvantageHead(observation_dim=3, action_size=2)
    torch.nn.init.normal_(head.mlp[-1].weight)
    observations = torch.randn(2, 3)
    centering_actions = torch.randn(2, 4, 2)
    behavior_actions = centering_actions[:, 0]
    centered = head(observations, behavior_actions, centering_actions[:, :1])
    assert torch.allclose(centered, torch.zeros_like(centered))


def test_centered_head_rejects_mismatched_samples():
    head = DirectAdvantageHead(observation_dim=3, action_size=2)
    with pytest.raises(ValueError, match="every observation"):
        head(torch.zeros(2, 3), torch.zeros(2, 2), torch.zeros(1, 4, 2))


def test_discounted_macro_rewards_stop_at_terminal():
    rewards = torch.tensor([[1.0], [2.0], [8.0], [4.0]])
    terminals = torch.tensor([[False], [True], [False], [False]])
    chunk_rewards, chunk_terminals = discounted_macro_rewards(
        rewards, terminals, action_steps=4, discount=0.5
    )
    assert chunk_rewards.tolist() == [[2.0]]
    assert chunk_terminals.tolist() == [[True]]


def test_direct_advantage_residuals_bootstrap_and_terminal():
    advantages = torch.tensor([[0.2], [0.3], [0.4]])
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    terminals = torch.tensor([[False], [False], [True]])
    values = torch.tensor([[0.5], [0.6], [0.7]])
    residuals = direct_advantage_residuals(
        advantages,
        rewards,
        terminals,
        values,
        final_baseline_values=torch.tensor([0.8]),
        valid_chunks=torch.ones_like(terminals),
        macro_discount=0.5,
        horizon=2,
    )
    expected = torch.tensor(
        [
            -0.5 + (1.0 - 0.2) + 0.5 * (2.0 - 0.3) + 0.25 * 0.7,
            -0.6 + (2.0 - 0.3) + 0.5 * (3.0 - 0.4),
            -0.7 + (3.0 - 0.4),
        ]
    )
    assert torch.allclose(residuals, expected)


def test_direct_advantage_residuals_skip_incomplete_window():
    with pytest.raises(ValueError, match="no complete"):
        direct_advantage_residuals(
            advantages=torch.zeros(2, 1),
            macro_rewards=torch.zeros(2, 1),
            terminals=torch.zeros(2, 1, dtype=torch.bool),
            baseline_values=torch.zeros(2, 1),
            final_baseline_values=torch.zeros(1),
            valid_chunks=torch.tensor([[True], [False]]),
            macro_discount=0.9,
            horizon=2,
        )
