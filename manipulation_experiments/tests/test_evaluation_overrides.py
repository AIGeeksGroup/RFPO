import pytest

from src.evaluation_overrides import validate_action_steps


def test_validate_action_steps_accepts_receding_horizon_override():
    assert validate_action_steps(4, 16) == 4


@pytest.mark.parametrize("action_steps", [0, 17])
def test_validate_action_steps_rejects_out_of_horizon_values(action_steps):
    with pytest.raises(ValueError, match="action_steps must be in"):
        validate_action_steps(action_steps, 16)
