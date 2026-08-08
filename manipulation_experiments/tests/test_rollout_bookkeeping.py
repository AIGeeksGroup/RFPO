import torch

from src.rollout_bookkeeping import prepare_invalid_step_mask


def test_prepare_invalid_step_mask_clears_stale_entries_in_place():
    invalid_steps = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    storage_pointer = invalid_steps.data_ptr()

    prepare_invalid_step_mask(invalid_steps, reset_each_iteration=True)

    assert invalid_steps.data_ptr() == storage_pointer
    torch.testing.assert_close(invalid_steps, torch.zeros_like(invalid_steps))


def test_prepare_invalid_step_mask_preserves_official_behavior_when_disabled():
    invalid_steps = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected = invalid_steps.clone()

    prepare_invalid_step_mask(invalid_steps, reset_each_iteration=False)

    torch.testing.assert_close(invalid_steps, expected)

