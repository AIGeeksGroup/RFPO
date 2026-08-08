import torch

from src.rollout_bookkeeping import (
    apply_zero_sampling_mask,
    apply_source_sampling_scale,
    build_rollout_zero_sampling_mask,
    prepare_invalid_step_mask,
)


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


def test_build_rollout_zero_sampling_mask_uses_global_environment_ids():
    rank_zero = build_rollout_zero_sampling_mask(
        4, 0.3, global_num_envs=10, global_offset=0
    )
    rank_one = build_rollout_zero_sampling_mask(
        6, 0.3, global_num_envs=10, global_offset=4
    )

    torch.testing.assert_close(
        torch.cat((rank_zero, rank_one)),
        torch.tensor([True, True, True, False, False, False, False, False, False, False]),
    )


def test_apply_zero_sampling_mask_preserves_gaussian_rows_and_input():
    source = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    original = source.clone()

    mixed = apply_zero_sampling_mask(source, torch.tensor([True, False, True]))

    torch.testing.assert_close(source, original)
    torch.testing.assert_close(mixed[0], torch.zeros_like(mixed[0]))
    torch.testing.assert_close(mixed[1], source[1])
    torch.testing.assert_close(mixed[2], torch.zeros_like(mixed[2]))


def test_rollout_zero_sampling_fraction_must_be_valid():
    try:
        build_rollout_zero_sampling_mask(4, 1.1)
    except ValueError as exc:
        assert "zero_fraction" in str(exc)
    else:
        raise AssertionError("invalid zero fraction should raise ValueError")


def test_apply_source_sampling_scale_scales_each_batch_row():
    source = torch.ones(3, 2, 4)

    scaled = apply_source_sampling_scale(source, torch.tensor([0.5, 1.0, 0.0]))

    torch.testing.assert_close(scaled[0], torch.full_like(scaled[0], 0.5))
    torch.testing.assert_close(scaled[1], torch.ones_like(scaled[1]))
    torch.testing.assert_close(scaled[2], torch.zeros_like(scaled[2]))
    torch.testing.assert_close(source, torch.ones_like(source))
