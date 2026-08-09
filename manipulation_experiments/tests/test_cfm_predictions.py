import torch

from src.cfm_predictions import flow_endpoint_predictions


def test_velocity_parameterization_recovers_both_endpoints():
    actions = torch.tensor([[[1.0], [3.0]]])
    noise = torch.tensor([[[5.0], [7.0]]])
    times = torch.tensor([[[0.25]]])
    velocity = torch.full_like(actions, 2.0)
    x_t = (1 - times) * actions + times * noise

    returned_velocity, x0_pred, x1_pred = flow_endpoint_predictions(
        x_t, times, velocity, output_parameterization="u"
    )

    expected_x0 = x_t - times * 2.0
    assert torch.equal(returned_velocity, velocity)
    assert torch.allclose(x0_pred, expected_x0)
    assert torch.allclose(x1_pred, expected_x0 + 2.0)


def test_x0_parameterization_is_algebraically_consistent():
    x_t = torch.tensor([[[2.0], [4.0]]])
    times = torch.tensor([[[0.25]]])
    x0 = torch.tensor([[[1.0], [3.0]]])

    velocity, x0_pred, x1_pred = flow_endpoint_predictions(
        x_t, times, x0, output_parameterization="x0"
    )

    assert torch.equal(x0_pred, x0)
    assert torch.allclose(velocity, torch.full_like(x0, 4.0))
    assert torch.allclose(x1_pred, torch.tensor([[[5.0], [7.0]]]))
