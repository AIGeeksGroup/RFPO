import pytest
import torch

from src.source_priors import (
    apply_previous_action_prior,
    split_action_history,
    update_ar1_gaussian_source,
)


def test_previous_action_prior_warms_only_available_prefixes():
    source_noise = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
    previous_actions = torch.tensor(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[50.0, 60.0], [70.0, 80.0]],
        ]
    )

    result = apply_previous_action_prior(
        source_noise,
        previous_actions,
        has_previous_actions=torch.tensor([True, False]),
        sigma=0.5,
    )

    torch.testing.assert_close(result[0, :2], previous_actions[0] + 0.5 * source_noise[0, :2])
    torch.testing.assert_close(result[0, 2:], source_noise[0, 2:])
    torch.testing.assert_close(result[1], source_noise[1])


def test_previous_action_prior_rejects_negative_sigma():
    with pytest.raises(ValueError, match="non-negative"):
        apply_previous_action_prior(
            torch.zeros(1, 2, 1),
            torch.zeros(1, 1, 1),
            torch.ones(1, dtype=torch.bool),
            sigma=-0.1,
        )


def test_ar1_source_preserves_first_innovation_without_aliasing():
    innovation = torch.tensor([[1.0, -2.0]])

    source = update_ar1_gaussian_source(innovation, None, 0.9)

    torch.testing.assert_close(source, innovation)
    assert source.data_ptr() != innovation.data_ptr()


def test_ar1_source_uses_stationary_innovation_scale():
    previous = torch.tensor([[2.0, -1.0]])
    innovation = torch.tensor([[1.0, 3.0]])

    source = update_ar1_gaussian_source(innovation, previous, 0.6)

    torch.testing.assert_close(source, 0.6 * previous + 0.8 * innovation)


def test_ar1_source_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="correlation"):
        update_ar1_gaussian_source(torch.zeros(2), None, 1.0)
    with pytest.raises(ValueError, match="identical shapes"):
        update_ar1_gaussian_source(torch.zeros(2), torch.zeros(3), 0.9)


def test_split_action_history_falls_back_at_episode_boundary():
    actions = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6, 1)
    padding = torch.tensor(
        [
            [False, False, False, False, False, False],
            [True, False, False, False, False, False],
        ]
    )

    previous, target, target_padding, available = split_action_history(
        actions, padding, history_steps=2, horizon=4
    )

    torch.testing.assert_close(previous, actions[:, :2])
    torch.testing.assert_close(target, actions[:, 2:])
    torch.testing.assert_close(target_padding, padding[:, 2:])
    assert available.tolist() == [True, False]
