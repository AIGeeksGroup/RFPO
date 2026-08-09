import numpy as np
import pytest
import torch

from src.binary_gripper_projection import (
    analyze_binary_gripper_projection,
    project_binary_gripper,
)


def test_projection_changes_only_final_coordinate_without_mutating_input():
    actions = torch.tensor([[0.25, -0.5, -0.01], [-0.75, 0.5, 0.0]])
    original = actions.clone()

    projected = project_binary_gripper(actions)

    assert torch.equal(actions, original)
    assert torch.equal(projected[:, :-1], actions[:, :-1])
    assert torch.equal(projected[:, -1], torch.tensor([-1.0, 1.0]))


@pytest.mark.parametrize("actions", [torch.empty(2, 0), torch.tensor([[float("nan")]])])
def test_projection_rejects_invalid_actions(actions):
    with pytest.raises(ValueError):
        project_binary_gripper(actions)


def test_projection_audit_passes_locked_reward_gate():
    raw = np.tile(np.array([[0.1, -0.2, 0.25]]), (40, 1))
    projected = raw.copy()
    projected[:, -1] = 1.0
    hashes = [f"seed-{index}" for index in range(20)]

    analysis = analyze_binary_gripper_projection(
        raw,
        projected,
        [1] * 17 + [0] * 3,
        hashes,
        hashes,
        audit_mode=True,
    )

    assert analysis["arm_coordinates_bitwise_exact"]
    assert analysis["projected_gripper_support_exact"]
    assert analysis["materially_changed_gripper_count"] == 40
    assert analysis["initial_observation_hashes_exact"]
    assert analysis["passed"]


def test_projection_audit_rejects_arm_change_hash_mismatch_and_low_reward():
    raw = np.tile(np.array([[0.1, -0.2, 0.25]]), (40, 1))
    projected = raw.copy()
    projected[:, 0] += 1e-12
    projected[:, -1] = 1.0
    hashes = [f"seed-{index}" for index in range(20)]
    control_hashes = list(hashes)
    control_hashes[-1] = "mismatch"

    analysis = analyze_binary_gripper_projection(
        raw,
        projected,
        [1] * 16 + [0] * 4,
        hashes,
        control_hashes,
        audit_mode=True,
    )

    assert not analysis["arm_coordinates_bitwise_exact"]
    assert not analysis["initial_observation_hashes_exact"]
    assert not analysis["validity_passed"]
    assert not analysis["passed"]
