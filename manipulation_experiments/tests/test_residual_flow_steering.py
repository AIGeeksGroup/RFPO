import pytest
import torch
from torch import nn

from src.residual_flow_steering import (
    ResidualFlowSteeringPolicy,
    assert_frozen_parameters_unchanged,
    clipped_ppo_loss,
    temporal_clipped_ppo_loss,
)


def test_rfs_initialization_preserves_gaussian_source_and_small_residual() -> None:
    torch.manual_seed(3)
    policy = ResidualFlowSteeringPolicy(11, horizon=4, action_dim=3)
    observations = torch.randn(2048, 11)

    sampled = policy.sample(observations)
    deterministic = policy.sample(observations[:2], deterministic=True)

    assert sampled.latent.shape == (2048, 4, 3)
    assert sampled.residual.shape == (2048, 4, 3)
    assert sampled.latent.mean().item() == pytest.approx(0.0, abs=0.03)
    assert sampled.latent.std().item() == pytest.approx(1.0, abs=0.03)
    assert sampled.residual.abs().max().item() <= 0.1
    assert torch.equal(deterministic.latent, torch.zeros_like(deterministic.latent))
    assert torch.equal(deterministic.residual, torch.zeros_like(deterministic.residual))


def test_rfs_log_probability_recomputes_for_ppo() -> None:
    torch.manual_seed(7)
    policy = ResidualFlowSteeringPolicy(5, horizon=2, action_dim=2)
    observations = torch.randn(8, 5)
    action = policy.sample(observations)

    reevaluated = policy.evaluate_actions(
        observations, action.latent_raw, action.residual_raw
    )

    assert torch.allclose(action.log_prob, reevaluated.log_prob)
    loss, clip_fraction = clipped_ppo_loss(
        reevaluated.log_prob, action.log_prob, torch.randn(8), clip_coef=0.2
    )
    assert torch.isfinite(loss)
    assert clip_fraction.item() == 0.0


def test_frozen_parameter_guard_detects_changes() -> None:
    module = nn.Linear(3, 2)
    reference = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    assert_frozen_parameters_unchanged(module, reference)
    with torch.no_grad():
        module.weight[0, 0] += 1
    with pytest.raises(RuntimeError, match="parameters changed"):
        assert_frozen_parameters_unchanged(module, reference)


def test_temporal_and_joint_ppo_have_same_on_policy_gradient() -> None:
    torch.manual_seed(11)
    policy = ResidualFlowSteeringPolicy(5, horizon=4, action_dim=3)
    observations = torch.randn(9, 5)
    sampled = policy.sample(observations)
    advantages = torch.randn(9)

    joint = policy.evaluate_actions(observations, sampled.latent_raw, sampled.residual_raw)
    joint_loss, _ = clipped_ppo_loss(
        joint.log_prob, sampled.log_prob.detach(), advantages, clip_coef=0.2
    )
    joint_gradients = torch.autograd.grad(joint_loss, tuple(policy.parameters()))

    temporal = policy.evaluate_actions(observations, sampled.latent_raw, sampled.residual_raw)
    temporal_loss, _ = temporal_clipped_ppo_loss(
        temporal.log_prob_steps,
        sampled.log_prob_steps.detach(),
        advantages,
        clip_coef=0.2,
    )
    temporal_gradients = torch.autograd.grad(temporal_loss, tuple(policy.parameters()))

    for joint_gradient, temporal_gradient in zip(joint_gradients, temporal_gradients, strict=True):
        assert torch.allclose(joint_gradient, temporal_gradient, atol=2e-5, rtol=2e-5)
