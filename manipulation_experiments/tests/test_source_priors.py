import pytest
import torch

from src.source_priors import apply_previous_action_prior


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
