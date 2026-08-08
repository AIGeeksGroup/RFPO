import torch

from src.flow_integrators import explicit_midpoint_step


def test_explicit_midpoint_is_more_accurate_than_euler_for_linear_ode():
    state = torch.tensor([1.0])
    time = torch.tensor(0.0)
    dt = torch.tensor(0.5)

    def velocity(value: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return value

    euler = state + dt * velocity(state, time)
    midpoint = explicit_midpoint_step(state, time, dt, velocity)
    exact = state * torch.exp(dt)

    assert torch.abs(midpoint - exact) < torch.abs(euler - exact)


def test_explicit_midpoint_uses_exactly_two_velocity_evaluations():
    calls = []

    def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        calls.append((state.clone(), time.clone()))
        return torch.ones_like(state)

    result = explicit_midpoint_step(
        torch.tensor([2.0]), torch.tensor(1.0), torch.tensor(-0.25), velocity
    )

    assert len(calls) == 2
    assert torch.equal(calls[0][0], torch.tensor([2.0]))
    assert torch.equal(calls[0][1], torch.tensor(1.0))
    assert torch.equal(calls[1][0], torch.tensor([1.875]))
    assert torch.equal(calls[1][1], torch.tensor(0.875))
    assert torch.equal(result, torch.tensor([1.75]))
