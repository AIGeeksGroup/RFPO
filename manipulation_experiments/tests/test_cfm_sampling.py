import pytest
import torch

from src.cfm_sampling import sample_cfm_variables


def _generators(seed: int = 7):
    return torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 1)


def test_joint_antithetic_samples_are_exact_pairs():
    time_generator, noise_generator = _generators()
    times, noises = sample_cfm_variables(
        batch_size=2,
        num_samples=8,
        horizon=3,
        action_dim=2,
        mode="joint_antithetic",
        time_generator=time_generator,
        noise_generator=noise_generator,
        device="cpu",
    )
    times = times.reshape(2, 8, 1, 1)
    noises = noises.reshape(2, 8, 3, 2)

    torch.testing.assert_close(times[:, :4] + times[:, 4:], torch.ones(2, 4, 1, 1))
    torch.testing.assert_close(noises[:, :4] + noises[:, 4:], torch.zeros(2, 4, 3, 2))


def test_iid_samples_have_requested_shapes():
    time_generator, noise_generator = _generators()
    times, noises = sample_cfm_variables(
        batch_size=3,
        num_samples=5,
        horizon=4,
        action_dim=2,
        mode="iid",
        time_generator=time_generator,
        noise_generator=noise_generator,
        device="cpu",
    )

    assert times.shape == (15, 1, 1)
    assert noises.shape == (15, 4, 2)
    assert ((times >= 0.0) & (times < 1.0)).all()


def test_joint_antithetic_rejects_odd_sample_count():
    time_generator, noise_generator = _generators()
    with pytest.raises(ValueError, match="even"):
        sample_cfm_variables(
            batch_size=2,
            num_samples=7,
            horizon=3,
            action_dim=2,
            mode="joint_antithetic",
            time_generator=time_generator,
            noise_generator=noise_generator,
            device="cpu",
        )
