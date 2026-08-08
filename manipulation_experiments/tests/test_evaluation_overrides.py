import pytest

from src.evaluation_overrides import (
    completed_environment_success_rates,
    validate_action_steps,
)


def test_validate_action_steps_accepts_receding_horizon_override():
    assert validate_action_steps(4, 16) == 4


@pytest.mark.parametrize("action_steps", [0, 17])
def test_validate_action_steps_rejects_out_of_horizon_values(action_steps):
    with pytest.raises(ValueError, match="action_steps must be in"):
        validate_action_steps(action_steps, 16)


def test_success_rates_skip_environments_without_completed_episodes():
    rates = completed_environment_success_rates(
        successes_per_env=[1, 0, 2, 0],
        done_episodes_per_env=[1, 0, 3, 0],
    )

    assert rates == pytest.approx([1.0, 2 / 3])


def test_success_rates_reject_mismatched_environment_counts():
    with pytest.raises(ValueError, match="same length"):
        completed_environment_success_rates([1], [1, 0])
