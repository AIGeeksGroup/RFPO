import numpy as np
import pytest
import torch
from torch import nn

from src.parameter_space_es import (
    PERTURBED_PARAMETER_NAMES,
    anchor_restored_exactly,
    apply_mirrored_direction,
    batched_observation_hashes,
    displacement_manifest,
    evaluate_signal,
    gaussian_direction,
    half_paired_rms,
    snapshot_parameters,
    validate_parameter_family,
)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.mlp = nn.Sequential(
            nn.Linear(4, 5), nn.Mish(),
            nn.Linear(5, 6), nn.Mish(),
            nn.Linear(6, 7), nn.Mish(),
            nn.Linear(7, 2),
        )
        self.model.frozen = nn.Linear(2, 2)


def test_locked_direction_is_deterministic_normalized_and_mirrored():
    policy = TinyPolicy()
    validate_parameter_family(policy)
    anchor = snapshot_parameters(policy)
    first = gaussian_direction(anchor, seed=20261000)
    second = gaussian_direction(anchor, seed=20261000)
    assert all(torch.equal(first[name], second[name]) for name in first)

    apply_mirrored_direction(policy, anchor, first, sign=1)
    positive = snapshot_parameters(policy)
    manifest = displacement_manifest(policy, anchor)
    assert manifest["finite"]
    assert manifest["unlisted_parameters_bitwise_unchanged"]
    assert set(manifest["relative_norms"]) == set(PERTURBED_PARAMETER_NAMES)
    assert all(value == pytest.approx(0.01, abs=1e-7) for value in manifest["relative_norms"].values())

    apply_mirrored_direction(policy, anchor, first, sign=-1)
    negative = snapshot_parameters(policy)
    for name in PERTURBED_PARAMETER_NAMES:
        torch.testing.assert_close(
            positive[name] - anchor[name],
            anchor[name] - negative[name],
            rtol=0,
            atol=torch.finfo(anchor[name].dtype).eps,
        )
    restore_delta = gaussian_direction(anchor, seed=20261001)
    apply_mirrored_direction(policy, anchor, restore_delta, sign=1)
    from src.parameter_space_es import restore_anchor
    restore_anchor(policy, anchor)
    assert anchor_restored_exactly(policy, anchor)


def test_validate_parameter_family_rejects_missing_locked_layer():
    policy = TinyPolicy()
    policy.model.mlp = policy.model.mlp[:-2]
    with pytest.raises(ValueError, match="missing locked ES parameters"):
        validate_parameter_family(policy)


def test_half_paired_rms_uses_half_difference():
    positive = np.array([[1.0, 3.0], [5.0, 7.0]])
    negative = np.array([[-1.0, 1.0], [3.0, 5.0]])
    assert half_paired_rms(positive, negative) == pytest.approx(1.0)


def test_batched_observation_hashes_are_order_invariant_and_sample_specific():
    images = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    first = batched_observation_hashes({"image": images, "state": states})
    second = batched_observation_hashes({"state": states, "image": images})
    assert first == second
    assert len(first) == 2
    assert first[0] != first[1]


def test_signal_gates_pass_for_reproducible_directions():
    d_a = np.array([1, 1, 1, 1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    d_b = np.array([1, 1, 1, 1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    successes = np.zeros((16, 2, 8), dtype=np.float64)
    for index, (left, right) in enumerate(zip(d_a, d_b)):
        for replica, difference in enumerate((left, right)):
            seed_slice = slice(replica * 4, replica * 4 + 4)
            if difference > 0:
                successes[index, 1, seed_slice] = 1
            elif difference < 0:
                successes[index, 0, seed_slice] = 1
    analysis = evaluate_signal(successes)
    assert analysis["all_signal_gates_pass"]
    assert analysis["correlation"] == pytest.approx(1.0)


def test_signal_gates_fail_for_constant_or_unreproducible_signal():
    successes = np.zeros((16, 2, 8), dtype=np.float64)
    analysis = evaluate_signal(successes)
    assert not analysis["all_signal_gates_pass"]
    assert np.isnan(analysis["correlation"])
