import pytest
import torch

from isaaclab_fpo.mirrored_rollouts import select_rollout_source


def test_select_rollout_source_preserves_exact_pair():
    primary = torch.tensor([[1.0, -2.0], [3.5, -4.5]])
    secondary = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    assert select_rollout_source("random", primary, secondary) is primary
    assert torch.equal(select_rollout_source("negative_random", primary), -primary)
    assert select_rollout_source("secondary_random", primary, secondary) is secondary


def test_secondary_rollout_requires_source():
    with pytest.raises(ValueError, match="requires a secondary source"):
        select_rollout_source("secondary_random", torch.zeros(2))
