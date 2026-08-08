import pytest
import torch

from src.ratio_rollback import rollback_clipped_ratio_loss


def standard_ppo_loss(ratios, advantages, clip_coef):
    unclipped = -advantages * ratios
    clipped = -advantages * ratios.clamp(1.0 - clip_coef, 1.0 + clip_coef)
    return torch.maximum(unclipped, clipped).mean()


def test_matches_ppo_loss_and_gradient_at_behavior_policy():
    advantages = torch.tensor([[1.5], [-0.7]])
    control_ratios = torch.ones(2, 3, requires_grad=True)
    candidate_ratios = control_ratios.detach().clone().requires_grad_()

    control_loss = standard_ppo_loss(control_ratios, advantages, 0.02)
    candidate_loss, active = rollback_clipped_ratio_loss(
        candidate_ratios, advantages, 0.02, 0.3
    )
    control_gradient = torch.autograd.grad(control_loss, control_ratios)[0]
    candidate_gradient = torch.autograd.grad(candidate_loss, candidate_ratios)[0]

    torch.testing.assert_close(candidate_loss, control_loss)
    torch.testing.assert_close(candidate_gradient, control_gradient)
    assert not active.any()


def test_reverses_improving_out_of_bound_gradients():
    ratios = torch.tensor([[1.1], [0.9]], requires_grad=True)
    advantages = torch.tensor([[2.0], [-3.0]])

    loss, active = rollback_clipped_ratio_loss(ratios, advantages, 0.02, 0.3)
    gradient = torch.autograd.grad(loss, ratios)[0]

    assert active.all()
    assert gradient[0].item() > 0.0
    assert gradient[1].item() < 0.0


def test_keeps_nonimproving_out_of_bound_ratios_unclipped():
    ratios = torch.tensor([[0.9], [1.1]], requires_grad=True)
    advantages = torch.tensor([[2.0], [-3.0]])

    loss, active = rollback_clipped_ratio_loss(ratios, advantages, 0.02, 0.3)
    gradient = torch.autograd.grad(loss, ratios)[0]

    assert not active.any()
    torch.testing.assert_close(gradient, -advantages / advantages.numel())


@pytest.mark.parametrize("clip_coef,alpha", [(0.0, 0.3), (1.0, 0.3), (0.02, 0.0)])
def test_rejects_invalid_coefficients(clip_coef, alpha):
    with pytest.raises(ValueError):
        rollback_clipped_ratio_loss(torch.ones(1), torch.ones(1), clip_coef, alpha)
