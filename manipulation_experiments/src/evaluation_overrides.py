from __future__ import annotations


def validate_action_steps(action_steps: int, prediction_horizon: int) -> int:
    """Validate an evaluation-time receding-horizon action count."""
    if not 1 <= action_steps <= prediction_horizon:
        raise ValueError(
            f"action_steps must be in [1, {prediction_horizon}], got {action_steps}"
        )
    return action_steps
