#!/usr/bin/env python3
"""CPU smoke tests for the four reflow innovation extensions."""

from __future__ import annotations

import sys

import torch

from isaaclab_fpo.modules.actor_critic import ActorCritic
from isaaclab_fpo.modules.reflow_extensions import (
    StepPredictor,
    compute_advantage_reflow_weights,
    compute_discretization_gap,
    compute_path_straightness,
    map_gap_to_reflow_lambda,
)
from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg
from isaaclab_fpo.task_cfgs import GO2_FPO_VARIANTS


def test_reflow_extensions():
    adv = torch.tensor([1.0, -0.5, 0.3, 2.0])
    w_ra = compute_advantage_reflow_weights(adv, "reward_aware", threshold=0.0)
    w_fpo = compute_advantage_reflow_weights(adv, "fpo_operator", threshold=0.0)
    _assert_close(w_ra.sum().item(), 1.0)
    _assert_close(w_fpo.sum().item(), 1.0)

    x0 = torch.randn(4, 12)
    x1 = torch.randn(4, 12)
    u = x0 - x1 + 0.01 * torch.randn_like(x0)
    straight = compute_path_straightness(x0, x1, u)
    assert straight.shape == (4,)

    gap = compute_discretization_gap(x0, x1)
    assert gap.shape == (4,)

    assert abs(map_gap_to_reflow_lambda(0.0, 0.1, 1.0, 0.05, 0.4) - 0.1) < 1e-6
    assert abs(map_gap_to_reflow_lambda(1.0, 0.1, 1.0, 0.05, 0.4) - 1.0) < 1e-6

    pred = StepPredictor(num_obs=48, step_bins=[1, 4, 8, 16, 64])
    obs = torch.randn(8, 48)
    steps = pred.predict_steps(obs)
    assert steps.min() >= 1
    assert steps.max() <= 64


def _assert_close(a: float, b: float, tol: float = 1e-5):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_actor_critic_variants():
    obs_dim, critic_dim, act_dim = 48, 48, 12
    batch = 16
    obs = torch.randn(batch, obs_dim)
    adv = torch.randn(batch, 1)

    for name in (
        "reflow",
        "reflow_adaptive_lambda",
        "reflow_random_x0",
        "reflow_teacher_kd",
        "reward_aware",
        "adaptive_compute",
        "fpo_operator",
        "theory",
        "all_ideas",
        "all_ideas_teacher_kd",
    ):
        variant = GO2_FPO_VARIANTS[name]()
        cfg = variant.policy
        policy = ActorCritic(obs_dim, critic_dim, act_dim, cfg)
        policy.train()

        if variant.algorithm.reflow_enabled:
            mode = variant.algorithm.reflow_mode
            loss = policy.get_reflow_loss(
                obs,
                n_samples_per_obs=2,
                advantages=adv if mode in {"reward_aware", "fpo_operator"} else None,
                reflow_mode=mode,
            )
            assert torch.isfinite(loss).all(), f"{name} reflow loss not finite: {loss}"

        if variant.algorithm.adaptive_compute_enabled:
            loss, metrics = policy.get_adaptive_compute_loss(obs, advantages=adv)
            assert torch.isfinite(loss).all(), f"{name} adaptive loss not finite"
            assert "adaptive_mean_steps" in metrics

        if variant.algorithm.random_x0_consistency_enabled:
            loss, metrics = policy.get_random_x0_consistency_loss(
                obs,
                step_bins=variant.algorithm.random_x0_consistency_steps,
            )
            assert torch.isfinite(loss).all(), f"{name} random_x0 loss not finite"
            assert "random_x0_consistency_loss" in metrics
            # mix mode should produce both zero and random starts
            if cfg.train_flow_x0_mode == "mix":
                x0 = policy._sample_train_flow_x0(64, obs.device)
                assert x0.shape == (64, act_dim)

        if variant.algorithm.theory_metrics_enabled:
            metrics = policy.compute_theory_metrics(obs[:8])
            assert "path_straightness" in metrics
            assert any(k.startswith("discretization_gap_") for k in metrics)

        if variant.algorithm.reflow_adaptive_lambda_enabled:
            gap = policy.compute_random_x0_step_gap(obs[:8], low_steps=1)
            assert gap >= 0.0

        if variant.algorithm.teacher_kd_enabled:
            loss, metrics = policy.get_teacher_kd_loss(
                obs,
                obs,
                teacher_actor=policy.actor,
                step_bins=variant.algorithm.teacher_kd_steps,
                zero_x0_prob=variant.algorithm.teacher_aux_zero_x0_prob,
            )
            assert torch.isfinite(loss).all(), f"{name} teacher_kd loss not finite"
            assert "teacher_kd_loss" in metrics


def main():
    test_reflow_extensions()
    test_actor_critic_variants()
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
