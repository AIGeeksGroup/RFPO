import copy

import torch

from src.factorized_source import (
    analyze_factorized_source_audit,
    build_source_audit_records,
    build_stateless_gaussian_sources,
    zero_gripper_source,
)


def test_stateless_sources_repeat_by_environment_and_plan_key():
    kwargs = {
        "base_seed": 101,
        "environment_ids": [2, 5],
        "plan_indices": [3, 7],
        "horizon": 4,
        "action_dim": 7,
        "device": "cpu",
    }

    first = build_stateless_gaussian_sources(**kwargs)
    second = build_stateless_gaussian_sources(**kwargs)
    changed = build_stateless_gaussian_sources(**{**kwargs, "plan_indices": [3, 8]})

    assert torch.equal(first, second)
    assert torch.equal(first[0], changed[0])
    assert not torch.equal(first[1], changed[1])


def test_zero_gripper_source_preserves_arm_and_input_exactly():
    source = build_stateless_gaussian_sources(
        base_seed=9,
        environment_ids=[0, 1],
        plan_indices=[0, 0],
        horizon=3,
        action_dim=7,
        device="cpu",
    )
    original = source.clone()

    factorized = zero_gripper_source(source)

    assert torch.equal(source, original)
    assert torch.equal(factorized[..., :-1], source[..., :-1])
    assert torch.equal(factorized[..., -1], torch.zeros_like(factorized[..., -1]))


def _condition(source, successes):
    used = source
    return {
        "initial_observation_hashes": ["a", "b"],
        "first_actions": [[0.0, 0.0], [0.0, 0.0]],
        "episodes": [
            {"environment_seed": 10, "success": successes[0]},
            {"environment_seed": 11, "success": successes[1]},
        ],
        "source_records": build_source_audit_records(
            source,
            used,
            environment_ids=[0, 1],
            plan_indices=[0, 0],
        ),
    }


def test_factorized_source_smoke_validates_common_arm_noise_and_activity():
    raw = build_stateless_gaussian_sources(
        base_seed=17,
        environment_ids=[0, 1],
        plan_indices=[0, 0],
        horizon=4,
        action_dim=7,
        device="cpu",
    )
    control = _condition(raw, [0, 0])
    candidate = _condition(zero_gripper_source(raw), [1, 1])
    candidate["source_records"] = build_source_audit_records(
        raw,
        zero_gripper_source(raw),
        environment_ids=[0, 1],
        plan_indices=[0, 0],
    )
    candidate["first_actions"][0][-1] = 0.25

    analysis = analyze_factorized_source_audit(
        control, candidate, audit_mode=False
    )

    assert analysis["arm_source_hashes_exact"]
    assert analysis["candidate_gripper_source_zero"]
    assert analysis["initial_observation_hashes_exact"]
    assert analysis["validity_passed"]
    assert analysis["passed"]


def test_factorized_source_audit_requires_three_success_gain():
    raw = build_stateless_gaussian_sources(
        base_seed=23,
        environment_ids=[0, 1],
        plan_indices=[0, 0],
        horizon=4,
        action_dim=7,
        device="cpu",
    )
    control = _condition(raw, [1, 0])
    candidate = copy.deepcopy(control)
    candidate["source_records"] = build_source_audit_records(
        raw,
        zero_gripper_source(raw),
        environment_ids=[0, 1],
        plan_indices=[0, 0],
    )
    candidate["first_actions"][0][-1] = 0.25
    candidate["episodes"][1]["success"] = 1

    analysis = analyze_factorized_source_audit(
        control, candidate, audit_mode=True
    )

    assert analysis["success_gain"] == 1
    assert analysis["control_regime_valid"]
    assert analysis["validity_passed"]
    assert not analysis["passed"]
