import torch

from src.advantage_weighting import (
    clipped_mirror_ratio_loss,
    effective_sample_size,
    ess_softmax_weights,
)


def test_effective_sample_size_is_scale_invariant():
    weights = torch.tensor([1.0, 2.0, 3.0])

    torch.testing.assert_close(
        effective_sample_size(weights), effective_sample_size(weights * 7.0)
    )


def test_ess_softmax_weights_hit_target_and_have_mean_one():
    advantages = torch.linspace(-2.0, 2.0, 64).reshape(32, 2)

    weights, temperature, ess_fraction = ess_softmax_weights(advantages, 0.5)

    assert weights.shape == advantages.shape
    assert temperature.item() > 0.0
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(ess_fraction, torch.tensor(0.5), atol=1e-4, rtol=0.0)
    assert not weights.requires_grad


def test_ess_softmax_weights_are_uniform_for_equal_advantages():
    weights, _, ess_fraction = ess_softmax_weights(torch.ones(8), 0.5)

    torch.testing.assert_close(weights, torch.ones(8))
    torch.testing.assert_close(ess_fraction, torch.tensor(1.0))


def test_ess_softmax_weights_validate_fraction():
    try:
        ess_softmax_weights(torch.ones(8), 0.0)
    except ValueError as exc:
        assert "target_ess_fraction" in str(exc)
    else:
        raise AssertionError("invalid ESS fraction should raise ValueError")


def test_clipped_mirror_ratio_loss_broadcasts_and_clips_positive_updates():
    ratio = torch.tensor([[1.0, 1.2], [0.8, 1.0]], requires_grad=True)
    weights = torch.tensor([[2.0], [0.5]])

    loss = clipped_mirror_ratio_loss(ratio, weights, 0.1)
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(-1.275))
    torch.testing.assert_close(
        ratio.grad,
        torch.tensor([[-0.5, 0.0], [-0.125, -0.125]]),
    )
