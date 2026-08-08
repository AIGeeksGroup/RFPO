"""Small explicit integrators used by flow-policy inference."""

from collections.abc import Callable

from torch import Tensor


def explicit_midpoint_step(
    state: Tensor,
    time: Tensor,
    dt: Tensor,
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Advance one explicit midpoint step using exactly two velocity calls."""
    initial_velocity = velocity_fn(state, time)
    midpoint_state = state + 0.5 * dt * initial_velocity
    midpoint_velocity = velocity_fn(midpoint_state, time + 0.5 * dt)
    return state + dt * midpoint_velocity
