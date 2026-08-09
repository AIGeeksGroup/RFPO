from types import SimpleNamespace

import torch
from torch import nn

from isaaclab_fpo.modules.actor_critic import ActorCritic


def make_policy():
    cfg = SimpleNamespace(
        activation="elu",
        timestep_embed_dim=8,
        actor_mlp_output_scale=1.0,
        cfm_loss_t_inverse_cdf_beta=1.0,
        sampling_steps=64,
        integration_method="euler",
        cfm_loss_reduction="sqrt",
        actor_scale=1.0,
        action_perturb_std=0.0,
        training_sampling_steps=None,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        actor_final_layer_weight_scale=None,
    )
    return ActorCritic(3, 3, 2, cfg)


class StateVelocity(nn.Module):
    def forward(self, inputs):
        return inputs[..., -2:]


def integrate(policy, method, steps, source):
    observations = torch.zeros(source.shape[0], 3)
    times = torch.linspace(1.0, 0.0, steps + 1)
    fn = (
        policy._integrate_flow if method == "euler" else policy._integrate_flow_midpoint
    )
    return fn(observations, source, times[:-1], times[1:] - times[:-1], steps)


def test_midpoint_has_lower_error_than_equal_nfe_euler_for_linear_velocity():
    policy = make_policy()
    policy.actor = StateVelocity()
    source = torch.tensor([[1.0, -2.0]])

    euler = integrate(policy, "euler", 64, source.clone())
    midpoint = integrate(policy, "midpoint", 32, source.clone())
    exact = source * torch.exp(torch.tensor(-1.0))

    assert torch.mean((midpoint - exact) ** 2) < torch.mean((euler - exact) ** 2)


def test_explicit_source_shape_is_validated():
    policy = make_policy()
    observations = torch.zeros(4, 3)

    try:
        policy.act_inference(observations, source=torch.zeros(3, 2))
    except ValueError as error:
        assert "source must have shape" in str(error)
    else:
        raise AssertionError("expected invalid source shape to fail")
