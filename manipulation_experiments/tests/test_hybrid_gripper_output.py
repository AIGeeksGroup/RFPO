import torch

from src.factorized_source import (
    build_source_audit_records,
    build_stateless_gaussian_sources,
)
from src.hybrid_gripper_output import (
    analyze_hybrid_gripper_audit,
    build_hybrid_output_audit_records,
    substitute_deterministic_gripper,
)


def test_substitution_preserves_gaussian_arm_and_deterministic_gripper():
    gaussian = torch.randn(2, 4, 7)
    deterministic = torch.randn(2, 4, 7)
    original = gaussian.clone()

    hybrid = substitute_deterministic_gripper(gaussian, deterministic)

    assert torch.equal(gaussian, original)
    assert torch.equal(hybrid[..., :-1], gaussian[..., :-1])
    assert torch.equal(hybrid[..., -1], deterministic[..., -1])


def test_hybrid_audit_passes_three_success_gain_with_exact_sources():
    source = build_stateless_gaussian_sources(
        base_seed=5,
        environment_ids=[0, 1, 2, 3],
        plan_indices=[0, 0, 0, 0],
        horizon=4,
        action_dim=7,
        device="cpu",
    )
    gaussian = torch.randn(4, 4, 7)
    deterministic = torch.randn(4, 4, 7)
    hybrid = substitute_deterministic_gripper(gaussian, deterministic)
    source_records = build_source_audit_records(
        source, source, [0, 1, 2, 3], [0, 0, 0, 0]
    )
    seeds = [10, 11, 12, 13]
    control = {
        "initial_observation_hashes": [f"seed-{seed}" for seed in seeds],
        "first_actions": gaussian[:, 0].tolist(),
        "episodes": [
            {"environment_seed": seed, "success": int(index == 0)}
            for index, seed in enumerate(seeds)
        ],
        "source_records": source_records,
    }
    candidate = {
        "initial_observation_hashes": list(control["initial_observation_hashes"]),
        "first_actions": hybrid[:, 0].tolist(),
        "episodes": [
            {"environment_seed": seed, "success": 1}
            for seed in seeds
        ],
        "source_records": source_records,
        "hybrid_output_records": build_hybrid_output_audit_records(
            gaussian,
            deterministic,
            hybrid,
            [0, 1, 2, 3],
            [0, 0, 0, 0],
        ),
    }

    analysis = analyze_hybrid_gripper_audit(
        control, candidate, audit_mode=True
    )

    assert analysis["arm_source_hashes_exact"]
    assert analysis["arm_output_bitwise_exact"]
    assert analysis["gripper_output_bitwise_exact"]
    assert analysis["success_gain"] == 3
    assert analysis["passed"]
