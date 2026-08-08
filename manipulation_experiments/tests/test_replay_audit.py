import pytest
import torch

from src.replay_audit import effective_sample_fraction, successful_chunk_mask


def test_successful_chunk_mask_stops_at_episode_boundaries():
    rewards = torch.zeros(8, 2)
    dones = torch.zeros(8, 2)
    dones[4, 0] = 1
    rewards[2, 0] = 1
    rewards[7, 1] = 1

    mask = successful_chunk_mask(rewards, dones, n_action_steps=2)

    # Flattening matches [chunk, environment], as used by the FPO rollout tensors.
    assert mask.tolist() == [True, True, True, True, False, True, False, True]


def test_successful_chunk_mask_validates_chunk_length():
    with pytest.raises(ValueError, match="evenly divide"):
        successful_chunk_mask(torch.zeros(5, 2), torch.zeros(5, 2), 2)


def test_effective_sample_fraction():
    torch.testing.assert_close(effective_sample_fraction(torch.ones(8)), torch.tensor(1.0))
    torch.testing.assert_close(
        effective_sample_fraction(torch.tensor([1.0, 0.0, 0.0, 0.0])),
        torch.tensor(0.25),
    )
