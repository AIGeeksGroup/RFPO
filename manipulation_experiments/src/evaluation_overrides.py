from __future__ import annotations

from collections.abc import Sequence


def validate_action_steps(action_steps: int, prediction_horizon: int) -> int:
    """Validate an evaluation-time receding-horizon action count."""
    if not 1 <= action_steps <= prediction_horizon:
        raise ValueError(
            f"action_steps must be in [1, {prediction_horizon}], got {action_steps}"
        )
    return action_steps


def completed_environment_success_rates(
    successes_per_env: Sequence[int], done_episodes_per_env: Sequence[int]
) -> list[float]:
    """Return per-environment rates for environments with completed episodes."""
    if len(successes_per_env) != len(done_episodes_per_env):
        raise ValueError("success and episode counts must have the same length")

    return [
        successes / done_episodes
        for successes, done_episodes in zip(successes_per_env, done_episodes_per_env)
        if done_episodes > 0
    ]


def balanced_episode_quota(num_episodes: int, num_envs: int) -> int:
    """Return an equal per-environment episode quota for unbiased small screens."""
    if num_episodes < 1 or num_envs < 1:
        raise ValueError("episode and environment counts must be positive")
    if num_episodes % num_envs != 0:
        raise ValueError("balanced evaluation requires episodes divisible by environments")
    return num_episodes // num_envs
