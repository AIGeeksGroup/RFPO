import pytest
import torch

from src.reflow_targets import mix_reflow_targets


def test_mix_reflow_targets_selects_whole_samples_and_padding():
    dataset_actions = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    teacher_actions = dataset_actions + 100
    dataset_is_pad = torch.tensor([[False, True], [True, True], [False, False]])

    actions, is_pad = mix_reflow_targets(
        dataset_actions,
        dataset_is_pad,
        teacher_actions,
        use_teacher=torch.tensor([True, False, True]),
    )

    torch.testing.assert_close(actions[0], teacher_actions[0])
    torch.testing.assert_close(actions[1], dataset_actions[1])
    torch.testing.assert_close(actions[2], teacher_actions[2])
    assert is_pad.tolist() == [[False, False], [True, True], [False, False]]


def test_mix_reflow_targets_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        mix_reflow_targets(
            torch.zeros(2, 3, 1),
            torch.zeros(2, 3, dtype=torch.bool),
            torch.zeros(2, 2, 1),
            torch.ones(2, dtype=torch.bool),
        )

