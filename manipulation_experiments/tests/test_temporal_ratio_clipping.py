import torch

from src.temporal_ratio_clipping import (
    clipped_ratio_objective,
    positive_active_fraction,
)


def test_temporal_objective_matches_chunk_gradient_at_behavior_policy():
    old = torch.tensor(
        [[[0.2, 0.4], [0.1, 0.3]], [[0.5, 0.2], [0.3, 0.6]]]
    )
    advantages = torch.tensor([1.5, -0.7])
    valid = torch.ones(2, 2)

    chunk_current = old.clone().requires_grad_()
    _, chunk_loss = clipped_ratio_objective(
        old, chunk_current, advantages, valid, 0.02, temporal=False
    )
    chunk_gradient = torch.autograd.grad(chunk_loss, chunk_current)[0]

    temporal_current = old.clone().requires_grad_()
    _, temporal_loss = clipped_ratio_objective(
        old, temporal_current, advantages, valid, 0.02, temporal=True
    )
    temporal_gradient = torch.autograd.grad(temporal_loss, temporal_current)[0]

    torch.testing.assert_close(temporal_gradient, chunk_gradient)


def test_temporal_clipping_keeps_unshifted_positive_timestep_active():
    old = torch.zeros(1, 2, 1)
    current = torch.tensor([[[-0.10], [0.0]]], requires_grad=True)
    advantages = torch.ones(1)
    valid = torch.ones(1, 2)

    chunk_ratios, chunk_loss = clipped_ratio_objective(
        old, current, advantages, valid, 0.02, temporal=False
    )
    temporal_ratios, temporal_loss = clipped_ratio_objective(
        old, current, advantages, valid, 0.02, temporal=True
    )
    chunk_gradient = torch.autograd.grad(chunk_loss, current, retain_graph=True)[0]
    temporal_gradient = torch.autograd.grad(temporal_loss, current)[0]

    assert positive_active_fraction(chunk_ratios, advantages, valid, 0.02) == 0
    assert positive_active_fraction(temporal_ratios, advantages, valid, 0.02) == 0.5
    torch.testing.assert_close(chunk_gradient, torch.zeros_like(chunk_gradient))
    torch.testing.assert_close(temporal_gradient[0, 0], torch.tensor([0.0]))
    assert temporal_gradient[0, 1].abs().item() > 0


def test_invalid_timestep_has_no_temporal_gradient():
    old = torch.zeros(1, 2, 1)
    current = torch.zeros_like(old, requires_grad=True)
    advantages = torch.ones(1)
    valid = torch.tensor([[1.0, 0.0]])

    _, loss = clipped_ratio_objective(
        old, current, advantages, valid, 0.02, temporal=True
    )
    gradient = torch.autograd.grad(loss, current)[0]

    assert gradient[0, 0].abs().item() > 0
    torch.testing.assert_close(gradient[0, 1], torch.tensor([0.0]))
