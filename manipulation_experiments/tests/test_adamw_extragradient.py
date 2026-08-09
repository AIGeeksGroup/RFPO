import pytest
import torch
from torch import nn

from src.adamw_extragradient import (
    assign_flat_gradient,
    displacement_vector,
    fresh_adamw_step,
    parameters_equal_snapshot,
    restore_parameters,
    snapshot_parameters,
    trainable_parameters,
)


def test_fresh_step_matches_torch_adamw():
    control = nn.Linear(3, 2)
    candidate = nn.Linear(3, 2)
    candidate.load_state_dict(control.state_dict())
    gradient = torch.linspace(-0.3, 0.4, sum(p.numel() for p in control.parameters()))

    control_parameters = list(control.parameters())
    optimizer = torch.optim.AdamW(
        control_parameters, lr=1e-3, betas=(0.9, 0.99), eps=1e-5, weight_decay=1e-6
    )
    assign_flat_gradient(control_parameters, gradient)
    torch.nn.utils.clip_grad_norm_(control_parameters, 0.25)
    optimizer.step()

    fresh_adamw_step(
        list(candidate.parameters()),
        gradient,
        learning_rate=1e-3,
        betas=(0.9, 0.99),
        eps=1e-5,
        weight_decay=1e-6,
        max_grad_norm=0.25,
    )

    for expected, actual in zip(control.parameters(), candidate.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_snapshot_restore_and_displacement_are_exact():
    module = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    parameters = trainable_parameters(module)
    snapshot = snapshot_parameters(parameters)
    gradient = torch.ones(sum(parameter.numel() for parameter in parameters))
    fresh_adamw_step(
        parameters,
        gradient,
        learning_rate=1e-3,
        betas=(0.9, 0.99),
        eps=1e-5,
        weight_decay=1e-6,
        max_grad_norm=1.0,
    )
    assert displacement_vector(parameters, snapshot).norm().item() > 0
    assert not parameters_equal_snapshot(parameters, snapshot)

    restore_parameters(parameters, snapshot)
    assert parameters_equal_snapshot(parameters, snapshot)
    assert displacement_vector(parameters, snapshot).count_nonzero().item() == 0


def test_rejects_wrong_flat_gradient_size():
    parameters = list(nn.Linear(2, 1).parameters())
    with pytest.raises(ValueError, match="expected"):
        assign_flat_gradient(parameters, torch.ones(2))
