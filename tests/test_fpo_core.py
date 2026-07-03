import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


class FPOCoreTests(unittest.TestCase):
    def make_policy(self, **overrides):
        from algos.rl.fpo_core import FPOPolicyConfig, FPOStatePolicy

        values = dict(
            obs_dim=6,
            action_dim=3,
            actor_hidden_dims=(32, 16),
            critic_hidden_dims=(32, 16),
            timestep_embed_dim=8,
            sampling_steps=4,
            action_clip=1.0,
        )
        values.update(overrides)
        cfg = FPOPolicyConfig(**values)
        return FPOStatePolicy(cfg)

    def test_importing_fpo_core_does_not_require_sb3_or_gym(self):
        from algos.rl.fpo_core import FPOPolicyConfig

        self.assertEqual(FPOPolicyConfig(obs_dim=2, action_dim=1).obs_dim, 2)

    def test_zero_sampling_is_deterministic_and_stochastic_sampling_is_not(self):
        policy = self.make_policy()
        obs = torch.randn(4, 6)

        deterministic_a = policy.act(obs, deterministic=True)
        deterministic_b = policy.act(obs, deterministic=True)
        stochastic_a = policy.act(obs, deterministic=False)
        stochastic_b = policy.act(obs, deterministic=False)

        self.assertTrue(torch.allclose(deterministic_a, deterministic_b))
        self.assertFalse(torch.allclose(stochastic_a, stochastic_b))
        self.assertLessEqual(float(deterministic_a.detach().abs().max()), 1.0)

    def test_residual_base_blends_flow_action_with_base_policy(self):
        class BasePolicy:
            def predict(self, obs, deterministic=True):
                return torch.full((obs.shape[0], 3), 0.25).numpy(), None

        policy = self.make_policy(action_clip=10.0)
        obs = torch.randn(4, 6)
        flow_action = policy.flow_act(obs, deterministic=True)
        policy.set_residual_base(BasePolicy(), residual_coef=0.1)

        action = policy.act(obs, deterministic=True)

        expected = torch.full_like(flow_action, 0.25) + 0.1 * (flow_action - 0.25)
        self.assertTrue(torch.allclose(action, expected))

    def test_trainable_residual_base_uses_differentiable_policy_forward(self):
        class TorchPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.25))

            def forward(self, obs, deterministic=True):
                actions = self.weight.expand(obs.shape[0], 3)
                values = torch.zeros(obs.shape[0], 1, device=obs.device)
                log_prob = torch.zeros(obs.shape[0], device=obs.device)
                return actions, values, log_prob

        class BasePolicy:
            def __init__(self):
                self.policy = TorchPolicy()

            def predict(self, obs, deterministic=True):
                raise AssertionError("trainable base should not call numpy predict")

        base = BasePolicy()
        policy = self.make_policy(action_clip=10.0)
        obs = torch.randn(4, 6)
        policy.set_residual_base(base, residual_coef=0.0, trainable=True)

        action = policy.act(obs, deterministic=True)
        loss = action.sum()
        loss.backward()

        self.assertIsNotNone(base.policy.weight.grad)
        self.assertGreater(float(base.policy.weight.grad.abs()), 0.0)

    def test_actor_parameters_can_include_trainable_base_policy(self):
        class TorchPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.25))

            def forward(self, obs, deterministic=True):
                actions = self.weight.expand(obs.shape[0], 3)
                values = torch.zeros(obs.shape[0], 1, device=obs.device)
                log_prob = torch.zeros(obs.shape[0], device=obs.device)
                return actions, values, log_prob

        class BasePolicy:
            def __init__(self):
                self.policy = TorchPolicy()

        base = BasePolicy()
        policy = self.make_policy()
        policy.set_residual_base(base, residual_coef=0.0, trainable=True)

        params = list(policy.actor_parameters(include_trainable_base=True))

        self.assertIn(id(base.policy.weight), {id(param) for param in params})

    def test_residual_base_inverts_executed_action_for_cfm_space(self):
        class BasePolicy:
            def predict(self, obs, deterministic=True):
                return torch.full((obs.shape[0], 3), 0.25).numpy(), None

        policy = self.make_policy(action_clip=10.0)
        obs = torch.randn(4, 6)
        flow_action = policy.flow_act(obs, deterministic=True)
        policy.set_residual_base(BasePolicy(), residual_coef=0.2)
        executed_action = policy.act(obs, deterministic=True)

        recovered = policy.flow_actions_from_executed_actions(obs, executed_action)

        self.assertTrue(torch.allclose(recovered, flow_action, atol=1e-6))

    def test_flow_plus_mlp_residual_works_without_base_policy(self):
        policy = self.make_policy(
            action_head_mode="flow_plus_mlp_residual",
            residual_action_scale=0.2,
            action_clip=10.0,
        )
        final_layer = policy.residual_head[-1]
        with torch.no_grad():
            final_layer.bias.fill_(1.0)
        obs = torch.randn(4, 6)

        flow_action = policy.flow_act(obs, deterministic=True)
        action = policy.act(obs, deterministic=True)
        recovered = policy.flow_actions_from_executed_actions(obs, action)

        expected = flow_action + 0.2 * torch.tanh(torch.ones_like(flow_action))
        self.assertTrue(torch.allclose(action, expected, atol=1e-6))
        self.assertTrue(torch.allclose(recovered, flow_action, atol=1e-6))

    def test_mlp_residual_cfm_uses_executed_actions_directly(self):
        class BasePolicy:
            def predict(self, obs, deterministic=True):
                return torch.full((obs.shape[0], 3), 0.25).numpy(), None

        policy = self.make_policy(
            action_clip=10.0,
            action_head_mode="mlp_residual",
            residual_action_scale=0.5,
            residual_head_hidden_dims=(16,),
            residual_head_zero_init=True,
        )
        policy.set_residual_base(BasePolicy(), residual_coef=0.0)
        final_layer = policy.residual_head[-1]
        with torch.no_grad():
            final_layer.bias.fill_(1.0)
        obs = torch.randn(4, 6)
        executed_action = policy.act(obs, deterministic=True)

        recovered = policy.flow_actions_from_executed_actions(obs, executed_action)

        self.assertTrue(torch.allclose(recovered, executed_action, atol=1e-6))

    def test_flow_plus_mlp_residual_cfm_removes_residual_before_flow_inverse(self):
        class BasePolicy:
            def predict(self, obs, deterministic=True):
                return torch.full((obs.shape[0], 3), 0.25).numpy(), None

        policy = self.make_policy(
            action_clip=10.0,
            action_head_mode="flow_plus_mlp_residual",
            residual_action_scale=0.5,
            residual_head_hidden_dims=(16,),
            residual_head_zero_init=True,
        )
        policy.set_residual_base(BasePolicy(), residual_coef=0.2)
        final_layer = policy.residual_head[-1]
        with torch.no_grad():
            final_layer.bias.fill_(1.0)
        obs = torch.randn(4, 6)
        flow_action = policy.flow_act(obs, deterministic=True)
        executed_action = policy.act(obs, deterministic=True)

        recovered = policy.flow_actions_from_executed_actions(obs, executed_action)

        self.assertTrue(torch.allclose(recovered, flow_action, atol=1e-6))

    def test_ppo_base_only_head_matches_trainable_base_policy_action(self):
        class TorchPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(6, 3)

            def forward(self, obs, deterministic=True):
                actions = self.linear(obs)
                values = torch.zeros(obs.shape[0], 1, device=obs.device)
                log_prob = torch.zeros(obs.shape[0], device=obs.device)
                return actions, values, log_prob

        class BasePolicy:
            def __init__(self):
                self.policy = TorchPolicy()

        base = BasePolicy()
        policy = self.make_policy(action_clip=10.0, action_head_mode="ppo_base_only")
        policy.set_residual_base(base, residual_coef=0.0, trainable=True)
        obs = torch.randn(4, 6)

        action = policy.act(obs, deterministic=True)
        expected, _, _ = base.policy(obs, deterministic=True)

        self.assertTrue(torch.allclose(action, expected))

    def test_ppo_base_only_can_sample_through_base_policy_forward(self):
        class TorchPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.log_std = torch.nn.Parameter(torch.zeros(3))

            def forward(self, obs, deterministic=True):
                actions = torch.full((obs.shape[0], 3), 0.75, device=obs.device)
                values = torch.full((obs.shape[0], 1), 2.0, device=obs.device)
                log_prob = torch.full((obs.shape[0],), -3.0, device=obs.device)
                return actions, values, log_prob

        class BasePolicy:
            def __init__(self):
                self.policy = TorchPolicy()

        policy = self.make_policy(action_clip=1.0, action_head_mode="ppo_base_only")
        policy.set_residual_base(BasePolicy(), residual_coef=0.0, trainable=True)

        actions, log_prob, mean, raw_actions = policy.sample_gaussian_action(
            torch.randn(4, 6),
            action_std=0.5,
            return_raw_action=True,
            use_base_policy_forward=True,
        )

        self.assertTrue(torch.allclose(actions, torch.full_like(actions, 0.75)))
        self.assertTrue(torch.allclose(raw_actions, actions))
        self.assertTrue(torch.allclose(mean, actions))
        self.assertTrue(torch.allclose(log_prob, torch.full((4, 1), -3.0)))

    def test_ppo_base_only_can_evaluate_log_prob_through_base_policy(self):
        class TorchPolicy(torch.nn.Module):
            def evaluate_actions(self, obs, actions):
                values = torch.full((obs.shape[0], 1), 2.0, device=obs.device)
                log_prob = actions.sum(dim=-1)
                entropy = torch.full((obs.shape[0],), 0.5, device=obs.device)
                return values, log_prob, entropy

        class BasePolicy:
            def __init__(self):
                self.policy = TorchPolicy()

        policy = self.make_policy(action_clip=1.0, action_head_mode="ppo_base_only")
        policy.set_residual_base(BasePolicy(), residual_coef=0.0, trainable=True)
        actions = torch.ones(4, 3)

        log_prob, entropy = policy.action_log_prob(
            torch.randn(4, 6),
            actions,
            action_std=0.5,
            use_base_policy_evaluate=True,
        )

        self.assertTrue(torch.allclose(log_prob, torch.full((4, 1), 3.0)))
        self.assertTrue(torch.allclose(entropy, torch.full((4, 1), 0.5)))

    def test_action_distribution_mean_can_use_unclipped_ppo_base_raw_mean(self):
        class TorchPolicy(torch.nn.Module):
            def forward(self, obs, deterministic=True):
                actions = torch.full((obs.shape[0], 3), 2.0, device=obs.device)
                values = torch.zeros(obs.shape[0], 1, device=obs.device)
                log_prob = torch.zeros(obs.shape[0], device=obs.device)
                return actions, values, log_prob

        class BasePolicy:
            def __init__(self):
                self.policy = TorchPolicy()

        policy = self.make_policy(action_clip=1.0, action_head_mode="ppo_base_only")
        policy.set_residual_base(BasePolicy(), residual_coef=0.0, trainable=True)

        clipped = policy.act(torch.randn(4, 6), deterministic=True)
        raw_mean = policy.action_distribution_mean(
            torch.randn(4, 6),
            use_unclipped_base=True,
        )

        self.assertTrue(torch.allclose(clipped, torch.ones_like(clipped)))
        self.assertTrue(torch.allclose(raw_mean, torch.full_like(raw_mean, 2.0)))

    def test_mlp_residual_head_zero_init_starts_from_base_policy(self):
        class BasePolicy:
            def predict(self, obs, deterministic=True):
                return torch.full((obs.shape[0], 3), -0.2).numpy(), None

        policy = self.make_policy(
            action_clip=10.0,
            action_head_mode="mlp_residual",
            residual_action_scale=0.5,
            residual_head_hidden_dims=(16,),
            residual_head_zero_init=True,
        )
        policy.set_residual_base(BasePolicy(), residual_coef=1.0)
        obs = torch.randn(4, 6)

        action = policy.act(obs, deterministic=True)

        self.assertTrue(torch.allclose(action, torch.full_like(action, -0.2), atol=1e-6))

    def test_mlp_residual_head_changes_action_when_nonzero(self):
        class BasePolicy:
            def predict(self, obs, deterministic=True):
                return torch.zeros((obs.shape[0], 3)).numpy(), None

        policy = self.make_policy(
            action_clip=10.0,
            action_head_mode="mlp_residual",
            residual_action_scale=0.5,
            residual_head_hidden_dims=(16,),
            residual_head_zero_init=True,
        )
        policy.set_residual_base(BasePolicy(), residual_coef=1.0)
        final_layer = policy.residual_head[-1]
        with torch.no_grad():
            final_layer.bias.fill_(1.0)
        obs = torch.randn(4, 6)

        action = policy.act(obs, deterministic=True)

        self.assertTrue(torch.allclose(action, torch.full_like(action, 0.5 * torch.tanh(torch.tensor(1.0)))))

    def test_direct_mlp_head_controls_scratch_policy_without_flow_sampling(self):
        policy = self.make_policy(
            action_clip=10.0,
            action_head_mode="direct_mlp",
            direct_head_hidden_dims=(16,),
            direct_action_scale=0.5,
            direct_head_zero_init=True,
        )
        final_layer = policy.direct_action_head[-1]
        with torch.no_grad():
            final_layer.bias.fill_(1.0)
        obs = torch.randn(4, 6)

        deterministic_action = policy.act(obs, deterministic=True)
        stochastic_action = policy.act(obs, deterministic=False)

        expected = torch.full_like(deterministic_action, 0.5 * torch.tanh(torch.tensor(1.0)))
        self.assertTrue(torch.allclose(deterministic_action, expected))
        self.assertTrue(torch.allclose(stochastic_action, expected))

    def test_direct_mlp_gaussian_sampling_uses_direct_mean(self):
        policy = self.make_policy(
            action_clip=10.0,
            action_head_mode="direct_mlp",
            direct_head_hidden_dims=(16,),
            direct_action_scale=0.5,
            direct_head_zero_init=True,
        )
        final_layer = policy.direct_action_head[-1]
        with torch.no_grad():
            final_layer.bias.fill_(1.0)
        obs = torch.randn(4, 6)

        actions, log_prob, mean, raw_actions = policy.sample_gaussian_action(
            obs,
            action_std=0.2,
            return_raw_action=True,
        )

        expected_mean = torch.full_like(mean, 0.5 * torch.tanh(torch.tensor(1.0)))
        self.assertTrue(torch.allclose(mean, expected_mean))
        self.assertEqual(actions.shape, (4, 3))
        self.assertEqual(raw_actions.shape, (4, 3))
        self.assertEqual(log_prob.shape, (4, 1))

    def test_flow_plus_direct_residual_adds_direct_head_to_scratch_flow_action(self):
        policy = self.make_policy(
            action_clip=10.0,
            action_head_mode="flow_plus_direct_residual",
            direct_head_hidden_dims=(16,),
            direct_action_scale=0.4,
            direct_head_zero_init=True,
        )
        final_layer = policy.direct_action_head[-1]
        with torch.no_grad():
            final_layer.bias.fill_(1.0)
        obs = torch.randn(4, 6)

        flow_action = policy.flow_act(obs, deterministic=True)
        action = policy.act(obs, deterministic=True)

        expected_residual = torch.full_like(action, 0.4 * torch.tanh(torch.tensor(1.0)))
        self.assertTrue(torch.allclose(action, flow_action + expected_residual, atol=1e-6))

    def test_flow_plus_direct_residual_cfm_removes_direct_residual_before_flow_inverse(self):
        policy = self.make_policy(
            action_clip=10.0,
            action_head_mode="flow_plus_direct_residual",
            direct_head_hidden_dims=(16,),
            direct_action_scale=0.5,
            direct_head_zero_init=True,
        )
        final_layer = policy.direct_action_head[-1]
        with torch.no_grad():
            final_layer.bias.fill_(1.0)
        obs = torch.randn(2, 6)
        executed = torch.ones(2, 3)

        flow_actions = policy.flow_actions_from_executed_actions(obs, executed)

        expected_residual = 0.5 * torch.tanh(torch.tensor(1.0))
        self.assertTrue(torch.allclose(flow_actions, executed - expected_residual, atol=1e-6))

    def test_direct_mlp_actor_parameters_include_direct_head(self):
        policy = self.make_policy(
            action_head_mode="direct_mlp",
            direct_head_hidden_dims=(16,),
        )

        params = list(policy.actor_parameters())

        self.assertIn(id(policy.direct_action_head[-1].weight), {id(param) for param in params})

    def test_cfm_loss_supports_multiple_samples_per_action(self):
        policy = self.make_policy(cfm_loss_use_huber=True, cfm_loss_huber_delta=0.5)
        obs = torch.randn(5, 6)
        actions = torch.randn(5, 3).clamp(-1.0, 1.0)
        eps = torch.randn(5, 7, 3)
        t = torch.rand(5, 7, 1)

        loss, x1_pred, x0_pred = policy.get_cfm_loss(obs, actions, eps, t)

        self.assertEqual(loss.shape, (5, 7))
        self.assertEqual(x1_pred.shape, (5, 7, 3))
        self.assertEqual(x0_pred.shape, (5, 7, 3))
        self.assertFalse(torch.isnan(loss).any())

    def test_fpo_control_huber_style_matches_reference_scaling(self):
        policy = self.make_policy(cfm_loss_use_huber=True, cfm_loss_huber_delta=1.0, cfm_loss_huber_style="fpo_control")
        prediction = torch.tensor([[0.5, 2.0]])
        target = torch.zeros_like(prediction)

        loss = policy._reduce_error(prediction, target)

        expected = torch.tensor([(0.25 + 3.0) / 2.0])
        self.assertTrue(torch.allclose(loss, expected))

    def test_fpo_surrogate_uses_per_sample_cfm_ratios(self):
        from algos.rl.fpo_core import fpo_surrogate_loss

        advantages = torch.tensor([[1.0], [-1.0]])
        old_loss = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        new_loss = torch.tensor([[0.5, 2.5, 2.0], [4.0, 1.5, 1.0]])

        out = fpo_surrogate_loss(
            advantages=advantages,
            old_cfm_loss=old_loss,
            new_cfm_loss=new_loss,
            clip_range=0.2,
            cfm_diff_clip=5.0,
            trust_region_mode="ppo",
        )

        expected_ratio = torch.exp(old_loss - new_loss)
        self.assertEqual(out["ratio"].shape, old_loss.shape)
        self.assertTrue(torch.allclose(out["ratio"], expected_ratio))
        self.assertFalse(torch.isnan(out["surrogate_loss"]))

    def test_fpo_surrogate_can_average_cfm_ratios_over_samples(self):
        from algos.rl.fpo_core import fpo_surrogate_loss

        advantages = torch.tensor([[1.0], [-1.0]])
        old_loss = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        new_loss = torch.tensor([[0.0, 2.0], [3.0, 5.0]])

        out = fpo_surrogate_loss(
            advantages=advantages,
            old_cfm_loss=old_loss,
            new_cfm_loss=new_loss,
            clip_range=0.2,
            cfm_diff_clip=5.0,
            trust_region_mode="ppo",
            average_cfm_loss_over_samples=True,
        )

        expected_log_ratio = old_loss.mean(dim=-1, keepdim=True) - new_loss.mean(dim=-1, keepdim=True)
        self.assertEqual(out["ratio"].shape, (2, 1))
        self.assertTrue(torch.allclose(out["log_ratio"], expected_log_ratio))
        self.assertTrue(torch.allclose(out["ratio"], torch.exp(expected_log_ratio)))

    def test_fpo_surrogate_clamps_only_old_cfm_loss_like_reference(self):
        from algos.rl.fpo_core import fpo_surrogate_loss

        advantages = torch.tensor([[1.0]])
        old_loss = torch.tensor([[10.0]])
        new_loss = torch.tensor([[8.0]])

        out = fpo_surrogate_loss(
            advantages=advantages,
            old_cfm_loss=old_loss,
            new_cfm_loss=new_loss,
            clip_range=0.2,
            cfm_diff_clip=20.0,
            cfm_loss_clamp=4.0,
            trust_region_mode="ppo",
        )

        self.assertTrue(torch.allclose(out["log_ratio"], torch.tensor([[-4.0]])))

    def test_fpo_surrogate_can_scale_cfm_log_ratio(self):
        from algos.rl.fpo_core import fpo_surrogate_loss

        advantages = torch.tensor([[1.0]])
        old_loss = torch.tensor([[1.0, 2.0]])
        new_loss = torch.tensor([[0.5, 3.0]])

        out = fpo_surrogate_loss(
            advantages=advantages,
            old_cfm_loss=old_loss,
            new_cfm_loss=new_loss,
            clip_range=0.2,
            cfm_diff_clip=20.0,
            trust_region_mode="ppo",
            log_ratio_scale=10.0,
        )

        self.assertTrue(torch.allclose(out["ratio"], torch.exp(10.0 * (old_loss - new_loss))))

    def test_fpo_surrogate_supports_reference_upper_only_log_ratio_clamp(self):
        from algos.rl.fpo_core import fpo_surrogate_loss

        advantages = torch.ones(2, 1)
        old_loss = torch.tensor([[1.0, 1.0], [5.0, 5.0]])
        new_loss = torch.tensor([[4.0, 4.0], [1.0, 1.0]])

        out = fpo_surrogate_loss(
            advantages=advantages,
            old_cfm_loss=old_loss,
            new_cfm_loss=new_loss,
            clip_range=0.2,
            cfm_diff_clip=0.0,
            cfm_diff_clip_max=2.0,
            trust_region_mode="ppo",
        )

        expected_log_ratio = torch.tensor([[-3.0, -3.0], [2.0, 2.0]])
        self.assertTrue(torch.allclose(out["log_ratio"], expected_log_ratio))
        self.assertTrue(torch.allclose(out["ratio"], torch.exp(expected_log_ratio)))

    def test_fpo_surrogate_can_clamp_current_loss_for_negative_advantages(self):
        from algos.rl.fpo_core import fpo_surrogate_loss

        advantages = torch.tensor([[-1.0], [1.0]])
        old_loss = torch.zeros(2, 2)
        new_loss = torch.tensor([[10.0, 10.0], [10.0, 10.0]])

        out = fpo_surrogate_loss(
            advantages=advantages,
            old_cfm_loss=old_loss,
            new_cfm_loss=new_loss,
            clip_range=0.2,
            cfm_diff_clip=0.0,
            cfm_diff_clip_max=20.0,
            clamp_negative_advantage_loss=2.0,
            trust_region_mode="ppo",
        )

        self.assertTrue(torch.allclose(out["log_ratio"][0], torch.tensor([-2.0, -2.0])))
        self.assertTrue(torch.allclose(out["log_ratio"][1], torch.tensor([-10.0, -10.0])))

    def test_fpo_surrogate_can_ignore_negative_advantage_actor_updates(self):
        from algos.rl.fpo_core import fpo_surrogate_loss

        advantages = torch.tensor([[-2.0]])
        old_loss = torch.tensor([[1.0, 1.0]])
        new_loss = torch.tensor([[0.5, 0.5]])

        out = fpo_surrogate_loss(
            advantages=advantages,
            old_cfm_loss=old_loss,
            new_cfm_loss=new_loss,
            clip_range=0.2,
            cfm_diff_clip=5.0,
            trust_region_mode="ppo",
            positive_advantage_only=True,
        )

        self.assertAlmostEqual(float(out["surrogate_loss"]), 0.0)

    def test_gaussian_ppo_surrogate_matches_standard_clipped_objective(self):
        from algos.rl.fpo_core import gaussian_ppo_surrogate_loss

        advantages = torch.tensor([[1.0], [-1.0], [0.5]])
        old_log_prob = torch.tensor([[-0.2], [-0.3], [-0.4]])
        new_log_prob = torch.tensor([[-0.1], [-0.8], [-0.2]])

        out = gaussian_ppo_surrogate_loss(
            advantages=advantages,
            old_log_prob=old_log_prob,
            new_log_prob=new_log_prob,
            clip_range=0.2,
        )

        ratio = torch.exp(new_log_prob - old_log_prob)
        expected = torch.max(
            -advantages * ratio,
            -advantages * ratio.clamp(0.8, 1.2),
        ).mean()
        self.assertTrue(torch.allclose(out["surrogate_loss"], expected))
        self.assertTrue(torch.allclose(out["ratio"], ratio))

    def test_gaussian_sampling_uses_unclipped_action_for_log_prob(self):
        policy = self.make_policy(action_clip=1.0)
        obs = torch.zeros(1, 6)
        with patch("torch.distributions.Normal.rsample", return_value=torch.full((1, 3), 2.0)):
            actions, log_prob, mean, raw_actions = policy.sample_gaussian_action(
                obs,
                action_std=0.5,
                return_raw_action=True,
            )

        expected = torch.distributions.Normal(mean, torch.full_like(mean, 0.5)).log_prob(raw_actions).sum(
            dim=-1,
            keepdim=True,
        )
        clipped_expected = torch.distributions.Normal(mean, torch.full_like(mean, 0.5)).log_prob(actions).sum(
            dim=-1,
            keepdim=True,
        )
        self.assertTrue(torch.allclose(actions, torch.ones_like(actions)))
        self.assertTrue(torch.allclose(raw_actions, torch.full_like(raw_actions, 2.0)))
        self.assertTrue(torch.allclose(log_prob, expected))
        self.assertFalse(torch.allclose(log_prob, clipped_expected))

    def test_chunked_fpo_surrogate_sums_cfm_loss_over_valid_chunk_steps(self):
        from algos.rl.fpo_core import chunked_fpo_surrogate_loss

        advantages = torch.tensor([[1.0], [-1.0]])
        old_loss = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
                [[2.0, 2.0], [1.0, 1.0], [5.0, 5.0]],
            ]
        )
        new_loss = torch.tensor(
            [
                [[0.5, 1.5], [4.0, 5.0], [0.0, 0.0]],
                [[1.0, 1.0], [3.0, 3.0], [6.0, 6.0]],
            ]
        )
        valid_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

        out = chunked_fpo_surrogate_loss(
            advantages=advantages,
            old_cfm_loss=old_loss,
            new_cfm_loss=new_loss,
            valid_mask=valid_mask,
            clip_range=0.2,
            cfm_diff_clip=5.0,
            trust_region_mode="ppo",
            average_cfm_loss_in_chunk=False,
        )

        expected_old = torch.tensor([[4.0, 6.0], [2.0, 2.0]])
        expected_new = torch.tensor([[4.5, 6.5], [1.0, 1.0]])
        self.assertTrue(torch.allclose(out["old_chunk_cfm_loss"], expected_old))
        self.assertTrue(torch.allclose(out["new_chunk_cfm_loss"], expected_new))
        self.assertTrue(torch.allclose(out["ratio"], torch.exp((expected_old - expected_new).mean(dim=-1, keepdim=True))))

        scaled = chunked_fpo_surrogate_loss(
            advantages=advantages,
            old_cfm_loss=old_loss,
            new_cfm_loss=new_loss,
            valid_mask=valid_mask,
            clip_range=0.2,
            cfm_diff_clip=5.0,
            trust_region_mode="ppo",
            average_cfm_loss_in_chunk=False,
            log_ratio_scale=2.0,
        )
        self.assertTrue(torch.allclose(scaled["ratio"], torch.exp(2.0 * (expected_old - expected_new).mean(dim=-1, keepdim=True))))

    def test_chunked_fpo_surrogate_can_average_over_valid_chunk_steps(self):
        from algos.rl.fpo_core import chunked_fpo_surrogate_loss

        advantages = torch.tensor([[1.0]])
        old_loss = torch.tensor([[[2.0], [4.0], [100.0]]])
        new_loss = torch.tensor([[[1.0], [3.0], [0.0]]])
        valid_mask = torch.tensor([[1.0, 1.0, 0.0]])

        out = chunked_fpo_surrogate_loss(
            advantages=advantages,
            old_cfm_loss=old_loss,
            new_cfm_loss=new_loss,
            valid_mask=valid_mask,
            clip_range=0.2,
            cfm_diff_clip=5.0,
            average_cfm_loss_in_chunk=True,
        )

        self.assertTrue(torch.allclose(out["old_chunk_cfm_loss"], torch.tensor([[3.0]])))
        self.assertTrue(torch.allclose(out["new_chunk_cfm_loss"], torch.tensor([[2.0]])))

    def test_chunked_fpo_surrogate_uses_mean_valid_chunk_advantage(self):
        from algos.rl.fpo_core import chunked_fpo_surrogate_loss, fpo_surrogate_loss

        advantages = torch.tensor([[1.0, 3.0, 100.0]])
        old_loss = torch.tensor([[[2.0], [4.0], [100.0]]])
        new_loss = torch.tensor([[[1.0], [3.0], [0.0]]])
        valid_mask = torch.tensor([[1.0, 1.0, 0.0]])

        out = chunked_fpo_surrogate_loss(
            advantages=advantages,
            old_cfm_loss=old_loss,
            new_cfm_loss=new_loss,
            valid_mask=valid_mask,
            clip_range=0.2,
            cfm_diff_clip=5.0,
            average_cfm_loss_in_chunk=True,
        )
        expected = fpo_surrogate_loss(
            advantages=torch.tensor([[2.0]]),
            old_cfm_loss=torch.tensor([[3.0]]),
            new_cfm_loss=torch.tensor([[2.0]]),
            clip_range=0.2,
            cfm_diff_clip=5.0,
        )

        self.assertTrue(torch.allclose(out["surrogate_loss"], expected["surrogate_loss"]))

    def test_rollout_buffer_computes_normalized_advantages_and_minibatches(self):
        from algos.rl.fpo_rollout_buffer import FPOTransition, FPORolloutBuffer

        buffer = FPORolloutBuffer(
            num_envs=2,
            num_steps=4,
            obs_dim=6,
            action_dim=3,
            n_cfm_samples=5,
            device="cpu",
        )
        for step in range(4):
            buffer.add(
                FPOTransition(
                    obs=torch.full((2, 6), float(step)),
                    actions=torch.full((2, 3), 0.1 * step),
                    flow_actions=torch.full((2, 3), 1.0 + 0.1 * step),
                    rewards=torch.full((2,), 1.0),
                    dones=torch.zeros(2),
                    values=torch.zeros(2, 1),
                    old_cfm_loss=torch.zeros(2, 5),
                    old_x1_pred=torch.full((2, 5, 3), float(step)),
                    cfm_eps=torch.randn(2, 5, 3),
                    cfm_t=torch.rand(2, 5, 1),
                    old_log_prob=torch.full((2, 1), -0.5),
                )
            )

        buffer.compute_returns_and_advantages(
            last_values=torch.zeros(2, 1),
            gamma=0.95,
            gae_lambda=0.95,
            normalize_advantage=True,
        )
        batches = list(buffer.iter_minibatches(num_minibatches=3, num_epochs=1))

        self.assertEqual(len(batches), 3)
        self.assertEqual(sum(batch.obs.shape[0] for batch in batches), 8)
        self.assertTrue(all(batch.flow_actions.shape[1:] == (3,) for batch in batches))
        self.assertTrue(torch.allclose(torch.cat([b.flow_actions for b in batches]).sort(dim=0).values[0], torch.ones(3)))
        self.assertTrue(all(batch.old_x1_pred.shape[1:] == (5, 3) for batch in batches))
        self.assertTrue(all(batch.old_log_prob.shape[1:] == (1,) for batch in batches))
        self.assertAlmostEqual(float(buffer.advantages.mean()), 0.0, places=5)

    def test_rollout_buffer_keeps_policy_actions_separate_from_env_actions(self):
        from algos.rl.fpo_rollout_buffer import FPOTransition, FPORolloutBuffer

        buffer = FPORolloutBuffer(
            num_envs=1,
            num_steps=1,
            obs_dim=6,
            action_dim=3,
            n_cfm_samples=2,
            device="cpu",
        )
        buffer.add(
            FPOTransition(
                obs=torch.zeros(1, 6),
                actions=torch.ones(1, 3),
                flow_actions=torch.full((1, 3), 0.5),
                rewards=torch.ones(1),
                dones=torch.zeros(1),
                values=torch.zeros(1, 1),
                old_cfm_loss=torch.zeros(1, 2),
                old_x1_pred=torch.zeros(1, 2, 3),
                cfm_eps=torch.randn(1, 2, 3),
                cfm_t=torch.rand(1, 2, 1),
                policy_actions=torch.full((1, 3), 2.0),
            )
        )
        buffer.compute_returns_and_advantages(
            last_values=torch.zeros(1, 1),
            gamma=0.95,
            gae_lambda=0.95,
            normalize_advantage=False,
        )

        batch = next(buffer.iter_minibatches(num_minibatches=1, num_epochs=1))

        self.assertTrue(torch.allclose(batch.actions, torch.ones(1, 3)))
        self.assertTrue(torch.allclose(batch.policy_actions, torch.full((1, 3), 2.0)))
        self.assertTrue(torch.allclose(batch.flow_actions, torch.full((1, 3), 0.5)))

    def test_rollout_buffer_yields_valid_chunk_minibatches(self):
        from algos.rl.fpo_rollout_buffer import FPOTransition, FPORolloutBuffer

        buffer = FPORolloutBuffer(
            num_envs=2,
            num_steps=6,
            obs_dim=6,
            action_dim=3,
            n_cfm_samples=2,
            device="cpu",
        )
        for step in range(6):
            dones = torch.zeros(2)
            if step == 1:
                dones[0] = 1.0
            buffer.add(
                FPOTransition(
                    obs=torch.full((2, 6), float(step)),
                    actions=torch.full((2, 3), float(step)),
                    flow_actions=torch.full((2, 3), 10.0 + float(step)),
                    rewards=torch.ones(2),
                    dones=dones,
                    values=torch.zeros(2, 1),
                    old_cfm_loss=torch.full((2, 2), float(step)),
                    old_x1_pred=torch.full((2, 2, 3), 100.0 + float(step)),
                    cfm_eps=torch.randn(2, 2, 3),
                    cfm_t=torch.rand(2, 2, 1),
                )
            )
        buffer.compute_returns_and_advantages(
            last_values=torch.zeros(2, 1),
            gamma=0.95,
            gae_lambda=0.95,
            normalize_advantage=False,
        )

        batches = list(buffer.iter_chunk_minibatches(chunk_steps=3, num_minibatches=2, num_epochs=1))
        self.assertEqual(sum(batch.obs.shape[0] for batch in batches), 4)
        first = batches[0]
        self.assertEqual(first.obs.shape[1:], (3, 6))
        self.assertEqual(first.actions.shape[1:], (3, 3))
        self.assertEqual(first.flow_actions.shape[1:], (3, 3))
        self.assertEqual(first.old_cfm_loss.shape[1:], (3, 2))
        self.assertEqual(first.old_x1_pred.shape[1:], (3, 2, 3))
        self.assertEqual(first.cfm_eps.shape[1:], (3, 2, 3))
        masks = torch.cat([batch.valid_mask for batch in batches], dim=0)
        self.assertTrue((masks.sum(dim=1) >= 1.0).all())
        self.assertTrue((masks.sum(dim=1) < 3.0).any())
        self.assertTrue((masks == torch.tensor([[1.0, 1.0, 0.0]])).all(dim=1).any())

    def test_chunk_minibatches_share_flow_t_within_each_chunk(self):
        from algos.rl.fpo_rollout_buffer import FPOTransition, FPORolloutBuffer

        buffer = FPORolloutBuffer(
            num_envs=1,
            num_steps=4,
            obs_dim=6,
            action_dim=3,
            n_cfm_samples=2,
            device="cpu",
        )
        for step in range(4):
            buffer.add(
                FPOTransition(
                    obs=torch.full((1, 6), float(step)),
                    actions=torch.zeros(1, 3),
                    flow_actions=torch.full((1, 3), float(step)),
                    rewards=torch.ones(1),
                    dones=torch.zeros(1),
                    values=torch.zeros(1, 1),
                    old_cfm_loss=torch.zeros(1, 2),
                    old_x1_pred=torch.zeros(1, 2, 3),
                    cfm_eps=torch.randn(1, 2, 3),
                    cfm_t=torch.full((1, 2, 1), float(step + 1)),
                )
            )
        buffer.compute_returns_and_advantages(
            last_values=torch.zeros(1, 1),
            gamma=0.95,
            gae_lambda=0.95,
            normalize_advantage=False,
        )

        batches = list(buffer.iter_chunk_minibatches(chunk_steps=2, num_minibatches=1, num_epochs=1))
        cfm_t = torch.cat([batch.cfm_t for batch in batches], dim=0)

        self.assertTrue(torch.allclose(cfm_t[:, 0], cfm_t[:, 1]))

    def test_chunked_trainer_passes_full_chunk_advantages_to_surrogate(self):
        import inspect

        from algos.rl.fpo_trainer import FPOStateTrainer

        source = inspect.getsource(FPOStateTrainer._update_chunked)

        self.assertIn("chunk_advantages = self._precision_shaped_advantages(batch, batch.advantages).squeeze(-1)", source)
        self.assertNotIn("chunk_advantages = batch.advantages[:, :, 0]", source)

    def test_trainer_tracks_x1_pred_kl_for_fpo_plus_plus_update_size(self):
        import inspect

        from algos.rl.fpo_trainer import FPOStateTrainer

        collect_source = inspect.getsource(FPOStateTrainer.collect_rollouts)
        update_source = inspect.getsource(FPOStateTrainer.update)
        chunked_source = inspect.getsource(FPOStateTrainer._update_chunked)

        self.assertIn("old_x1_pred", collect_source)
        self.assertIn("x1_pred_kl", update_source)
        self.assertIn("x1_pred_kl", chunked_source)
        self.assertIn("_adjust_learning_rate_from_x1_kl", update_source)
        self.assertIn("_adjust_learning_rate_from_x1_kl", chunked_source)

    def test_object_precision_bonus_only_applies_in_target_stage(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = FPOStateTrainer.__new__(FPOStateTrainer)
        trainer.obj_precision_reward_coef = 2.0
        trainer.obj_precision_reward_target = 0.001
        trainer.obj_precision_reward_scale = 250.0
        trainer.obj_precision_reward_stage_min = 2.0
        trainer.obj_precision_reward_max = 2.0
        trainer.obj_precision_penalty_coef = 0.0
        trainer.obj_precision_penalty_target = 0.001
        trainer.obj_precision_penalty_power = 1.0
        trainer.obj_precision_penalty_max = 1.0
        trainer.obj_precision_delta_coef = 0.0
        trainer.device = torch.device("cpu")

        bonus = trainer._object_precision_bonus_from_infos(
            [
                {"stage": 1, "obj_com_err": 0.0},
                {"stage": 2, "obj_com_err": 0.001},
                {"stage": 2, "obj_com_err": 0.005},
                {},
            ]
        )

        self.assertIsNotNone(bonus)
        assert bonus is not None
        self.assertAlmostEqual(float(bonus[0]), 0.0)
        self.assertAlmostEqual(float(bonus[1]), 2.0)
        self.assertAlmostEqual(float(bonus[2]), float(2.0 * np.exp(-250.0 * 0.004)))
        self.assertAlmostEqual(float(bonus[3]), 0.0)

    def test_object_precision_penalty_continues_to_penalize_late_stage_error(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = FPOStateTrainer.__new__(FPOStateTrainer)
        trainer.obj_precision_reward_coef = 0.0
        trainer.obj_precision_reward_target = 0.001
        trainer.obj_precision_reward_scale = 250.0
        trainer.obj_precision_reward_stage_min = 2.0
        trainer.obj_precision_reward_max = 2.0
        trainer.obj_precision_penalty_coef = 250.0
        trainer.obj_precision_penalty_target = 0.001
        trainer.obj_precision_penalty_power = 1.0
        trainer.obj_precision_penalty_max = 1.0
        trainer.obj_precision_delta_coef = 0.0
        trainer.device = torch.device("cpu")

        bonus = trainer._object_precision_bonus_from_infos(
            [
                {"stage": 1, "obj_com_err": 0.01},
                {"stage": 2, "obj_com_err": 0.001},
                {"stage": 2, "obj_com_err": 0.003},
                {"stage": 2, "obj_com_err": 0.02},
            ]
        )

        self.assertIsNotNone(bonus)
        assert bonus is not None
        self.assertAlmostEqual(float(bonus[0]), 0.0)
        self.assertAlmostEqual(float(bonus[1]), 0.0)
        self.assertAlmostEqual(float(bonus[2]), -0.5)
        self.assertAlmostEqual(float(bonus[3]), -1.0)

    def test_object_precision_improvement_bonus_rewards_error_reduction(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = FPOStateTrainer.__new__(FPOStateTrainer)
        trainer.obj_precision_reward_coef = 0.0
        trainer.obj_precision_reward_target = 0.001
        trainer.obj_precision_reward_scale = 250.0
        trainer.obj_precision_reward_stage_min = 2.0
        trainer.obj_precision_reward_max = 2.0
        trainer.obj_precision_penalty_coef = 0.0
        trainer.obj_precision_penalty_target = 0.001
        trainer.obj_precision_penalty_power = 1.0
        trainer.obj_precision_penalty_max = 1.0
        trainer.obj_precision_delta_coef = 100.0
        trainer.obj_precision_delta_target = 0.0011
        trainer.obj_precision_delta_max = 0.5
        trainer.device = torch.device("cpu")
        trainer._last_obj_precision_err = torch.tensor([0.004, 0.002, 0.005])

        bonus = trainer._object_precision_bonus_from_infos(
            [
                {"stage": 2, "obj_com_err": 0.003},
                {"stage": 2, "obj_com_err": 0.003},
                {"stage": 1, "obj_com_err": 0.001},
            ]
        )

        self.assertIsNotNone(bonus)
        assert bonus is not None
        self.assertAlmostEqual(float(bonus[0]), 0.1, places=6)
        self.assertAlmostEqual(float(bonus[1]), -0.1, places=6)
        self.assertAlmostEqual(float(bonus[2]), 0.0, places=6)
        self.assertTrue(torch.allclose(trainer._last_obj_precision_err, torch.tensor([0.003, 0.003, 0.005])))

    def test_object_precision_improvement_state_resets_done_envs(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = FPOStateTrainer.__new__(FPOStateTrainer)
        trainer.obj_precision_reward_coef = 0.0
        trainer.obj_precision_reward_target = 0.001
        trainer.obj_precision_reward_scale = 250.0
        trainer.obj_precision_reward_stage_min = 2.0
        trainer.obj_precision_reward_max = 2.0
        trainer.obj_precision_penalty_coef = 0.0
        trainer.obj_precision_penalty_target = 0.001
        trainer.obj_precision_penalty_power = 1.0
        trainer.obj_precision_penalty_max = 1.0
        trainer.obj_precision_delta_coef = 100.0
        trainer.obj_precision_delta_target = 0.0011
        trainer.obj_precision_delta_max = 0.5
        trainer.device = torch.device("cpu")
        trainer._last_obj_precision_err = torch.tensor([0.004, 0.004])

        bonus = trainer._object_precision_bonus_from_infos(
            [
                {"stage": 2, "obj_com_err": 0.003},
                {"stage": 2, "obj_com_err": 0.003, "terminal_observation": np.zeros(1)},
            ]
        )

        self.assertIsNotNone(bonus)
        assert bonus is not None
        self.assertAlmostEqual(float(bonus[0]), 0.1, places=6)
        self.assertAlmostEqual(float(bonus[1]), 0.1, places=6)
        self.assertTrue(torch.isfinite(trainer._last_obj_precision_err[0]))
        self.assertFalse(torch.isfinite(trainer._last_obj_precision_err[1]))

    def test_object_precision_bonus_disabled_returns_none(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = FPOStateTrainer.__new__(FPOStateTrainer)
        trainer.obj_precision_reward_coef = 0.0
        trainer.obj_precision_penalty_coef = 0.0
        trainer.obj_precision_delta_coef = 0.0

        bonus = trainer._object_precision_bonus_from_infos([{"stage": 2, "obj_com_err": 0.001}])

        self.assertIsNone(bonus)

    def test_adaptive_x1_kl_preserves_optimizer_group_lr_scales(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        param_a = torch.nn.Parameter(torch.tensor(1.0))
        param_b = torch.nn.Parameter(torch.tensor(2.0))
        trainer = FPOStateTrainer.__new__(FPOStateTrainer)
        trainer.learning_rate = 1e-6
        trainer.lr_schedule = "adaptive"
        trainer.desired_x1_kl = 0.01
        trainer.min_learning_rate = 1e-7
        trainer.max_learning_rate = 1e-5
        trainer.optimizer = torch.optim.Adam(
            [
                {"params": [param_a], "lr": 1e-6},
                {"params": [param_b], "lr": 2.5e-7},
            ],
            lr=1e-6,
        )

        trainer._record_optimizer_lr_scales()
        trainer._adjust_learning_rate_from_x1_kl(torch.tensor(0.04))

        self.assertAlmostEqual(trainer.learning_rate, 1e-6 / 1.5)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], trainer.learning_rate)
        self.assertAlmostEqual(trainer.optimizer.param_groups[1]["lr"], trainer.learning_rate * 0.25)

        trainer._adjust_learning_rate_from_x1_kl(torch.tensor(0.001))

        self.assertAlmostEqual(trainer.learning_rate, 1e-6)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], trainer.learning_rate)
        self.assertAlmostEqual(trainer.optimizer.param_groups[1]["lr"], trainer.learning_rate * 0.25)

    def test_checkpoint_roundtrip_preserves_normalizers(self):
        from algos.rl.fpo_trainer import load_fpo_state_policy, save_fpo_checkpoint

        policy = self.make_policy()
        policy.set_normalizers(
            obs_mean=torch.arange(6, dtype=torch.float32),
            obs_std=torch.ones(6) * 2.0,
            action_mean=torch.ones(3) * 0.25,
            action_std=torch.ones(3) * 0.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "restore_checkpoint.pt")
            save_fpo_checkpoint(
                path=path,
                policy=policy,
                optimizer_state_dict=None,
                num_timesteps=123,
                extra={"current_stage": 1},
            )
            loaded = load_fpo_state_policy(path, device="cpu")

        obs = torch.randn(2, 6)
        self.assertTrue(torch.allclose(policy.act(obs, deterministic=True), loaded.policy.act(obs, deterministic=True)))
        self.assertEqual(loaded.num_timesteps, 123)
        self.assertEqual(loaded.extra["current_stage"], 1)

    def test_checkpoint_saves_trainable_base_policy_state(self):
        from algos.rl.fpo_trainer import save_fpo_checkpoint

        class TorchPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(6, 3)

            def forward(self, obs, deterministic=True):
                actions = self.linear(obs)
                values = torch.zeros(obs.shape[0], 1, device=obs.device)
                log_prob = torch.zeros(obs.shape[0], device=obs.device)
                return actions, values, log_prob

        class BasePolicy:
            def __init__(self):
                self.policy = TorchPolicy()

        policy = self.make_policy()
        policy.set_residual_base(BasePolicy(), residual_coef=0.0, trainable=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "restore_checkpoint.pt")
            save_fpo_checkpoint(
                path=path,
                policy=policy,
                optimizer_state_dict=None,
                num_timesteps=123,
            )
            checkpoint = torch.load(path, map_location="cpu")

        self.assertIsNotNone(checkpoint["base_policy_state_dict"])
        self.assertIn("linear.weight", checkpoint["base_policy_state_dict"])

    def test_evaluate_can_use_ppo_base_critic(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        class TorchPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.25))

            def forward(self, obs, deterministic=True):
                actions = self.weight.expand(obs.shape[0], 3)
                values = obs[:, :1] + 2.0
                log_prob = torch.zeros(obs.shape[0], device=obs.device)
                return actions, values, log_prob

            def evaluate_actions(self, obs, actions):
                values = obs[:, :1] + 2.0
                log_prob = torch.zeros(obs.shape[0], device=obs.device)
                entropy = torch.zeros(obs.shape[0], device=obs.device)
                return values, log_prob, entropy

        class BasePolicy:
            def __init__(self):
                self.policy = TorchPolicy()

        trainer = object.__new__(FPOStateTrainer)
        trainer.value_source = "ppo_base"
        trainer.ppo_base = BasePolicy()
        trainer.policy = self.make_policy()

        obs = torch.randn(4, 6)
        values = FPOStateTrainer.evaluate(trainer, obs)

        self.assertTrue(torch.allclose(values, obs[:, :1] + 2.0))

    def test_gaussian_action_std_can_use_ppo_base_log_std(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        class TorchPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.log_std = torch.nn.Parameter(torch.log(torch.tensor([0.1, 0.2, 0.3])))

        class BasePolicy:
            def __init__(self):
                self.policy = TorchPolicy()

        trainer = object.__new__(FPOStateTrainer)
        trainer.gaussian_action_std = "ppo_base"
        trainer.ppo_base = BasePolicy()

        action_std = FPOStateTrainer._current_action_std(trainer)

        self.assertTrue(torch.allclose(action_std.cpu(), torch.tensor([0.1, 0.2, 0.3])))

    def test_gaussian_action_std_from_ppo_base_supports_scaled_decay(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        class TorchPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.log_std = torch.nn.Parameter(torch.log(torch.tensor([0.1, 0.2, 0.3])))

        class BasePolicy:
            def __init__(self):
                self.policy = TorchPolicy()

        trainer = object.__new__(FPOStateTrainer)
        trainer.gaussian_action_std = "ppo_base"
        trainer.gaussian_action_std_scale = 0.5
        trainer.gaussian_action_std_min_scale = 0.25
        trainer.gaussian_action_std_decay_steps = 1000
        trainer.num_timesteps = 500
        trainer.ppo_base = BasePolicy()

        action_std = FPOStateTrainer._current_action_std(trainer)

        self.assertTrue(torch.allclose(action_std.cpu(), torch.tensor([0.0375, 0.075, 0.1125])))

    def test_gaussian_action_std_from_ppo_base_can_keep_log_std_trainable(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        class TorchPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.log_std = torch.nn.Parameter(torch.log(torch.tensor([0.1, 0.2, 0.3])))

        class BasePolicy:
            def __init__(self):
                self.policy = TorchPolicy()

        trainer = object.__new__(FPOStateTrainer)
        trainer.gaussian_action_std = "ppo_base"
        trainer.gaussian_action_std_scale = 1.0
        trainer.gaussian_action_std_min_scale = 0.0
        trainer.gaussian_action_std_decay_steps = 0
        trainer.gaussian_action_std_trainable = True
        trainer.num_timesteps = 0
        trainer.ppo_base = BasePolicy()

        action_std = FPOStateTrainer._current_action_std(trainer)
        action_std.sum().backward()

        self.assertTrue(action_std.requires_grad)
        self.assertIsNotNone(trainer.ppo_base.policy.log_std.grad)

    def test_fixed_gaussian_action_std_can_be_trainable(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.gaussian_action_std_source = "fixed"
        trainer.gaussian_action_std = 0.2
        trainer.gaussian_action_std_scale = 1.0
        trainer.gaussian_action_std_min_scale = 0.0
        trainer.gaussian_action_std_decay_steps = 0
        trainer.gaussian_log_std = torch.nn.Parameter(torch.log(torch.full((3,), 0.2)))
        trainer.num_timesteps = 0

        action_std = FPOStateTrainer._current_action_std(trainer)
        action_std.sum().backward()

        self.assertTrue(torch.allclose(action_std.detach().cpu(), torch.full((3,), 0.2)))
        self.assertTrue(action_std.requires_grad)
        self.assertIsNotNone(trainer.gaussian_log_std.grad)

    def test_optimizer_includes_trainable_fixed_gaussian_log_std(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.policy = self.make_policy()
        trainer.learning_rate = 1e-3
        trainer.gaussian_log_std = torch.nn.Parameter(torch.zeros(3))

        optimizer = FPOStateTrainer._make_optimizer(trainer)
        optimized_params = {id(param) for group in optimizer.param_groups for param in group["params"]}

        self.assertIn(id(trainer.gaussian_log_std), optimized_params)

    def test_optimizer_can_scale_residual_head_lr_independently(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.policy = self.make_policy()
        trainer.learning_rate = 1e-4
        trainer.flow_actor_lr_scale = 0.25
        trainer.residual_head_lr_scale = 8.0
        trainer.direct_head_lr_scale = 3.0
        trainer.gaussian_log_std = None

        optimizer = FPOStateTrainer._make_optimizer(trainer)

        lrs = [group["lr"] for group in optimizer.param_groups[:4]]
        for lr, expected in zip(lrs, [2.5e-5, 1e-4, 8e-4, 3e-4]):
            self.assertAlmostEqual(lr, expected)

    def test_policy_weight_loading_can_reset_direct_head_on_resume(self):
        from algos.rl.fpo_trainer import FPOStateTrainer, save_fpo_checkpoint

        checkpoint_policy = self.make_policy(
            action_head_mode="flow_plus_direct_residual",
            direct_head_zero_init=False,
        )
        with torch.no_grad():
            checkpoint_policy.direct_action_head[-1].weight.fill_(0.5)
            checkpoint_policy.direct_action_head[-1].bias.fill_(0.25)
        runtime_policy = self.make_policy(
            action_head_mode="flow_plus_direct_residual",
            direct_head_zero_init=False,
        )

        trainer = object.__new__(FPOStateTrainer)
        trainer.obs_dim = 6
        trainer.action_dim = 3
        trainer.device = torch.device("cpu")
        trainer.policy = runtime_policy
        trainer.current_stage = 0
        trainer.num_timesteps = 0
        trainer.reset_direct_head_on_resume = True
        trainer.gaussian_log_std = None
        trainer._make_optimizer = lambda: torch.optim.Adam(trainer.policy.parameters(), lr=1e-3)
        trainer._record_optimizer_lr_scales = lambda: None
        trainer.optimizer = trainer._make_optimizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "policy.pt")
            save_fpo_checkpoint(
                path=path,
                policy=checkpoint_policy,
                optimizer_state_dict=None,
                num_timesteps=123,
            )
            trainer.load_policy_weights(path, load_optimizer=False)

        final_layer = trainer.policy.direct_action_head[-1]
        self.assertTrue(torch.allclose(final_layer.weight, torch.zeros_like(final_layer.weight)))
        self.assertTrue(torch.allclose(final_layer.bias, torch.zeros_like(final_layer.bias)))

    def test_checkpoint_roundtrip_preserves_trainable_gaussian_log_std(self):
        import tempfile

        from algos.rl.fpo_trainer import FPOStateTrainer, load_fpo_state_policy

        trainer = object.__new__(FPOStateTrainer)
        trainer.policy = self.make_policy()
        trainer.optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=1e-3)
        trainer.num_timesteps = 123
        trainer.current_stage = 2
        trainer.iterations = 4
        trainer.cfg = SimpleNamespace(agent=SimpleNamespace(params=SimpleNamespace(
            gaussian_action_std_source="fixed",
            gaussian_action_std_trainable=True,
            gaussian_action_std=0.2,
            actor_objective="gaussian_ppo",
        )))
        trainer.gaussian_log_std = torch.nn.Parameter(torch.log(torch.tensor([0.11, 0.22, 0.33])))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.pt"
            FPOStateTrainer.save(trainer, str(path))

            loaded = load_fpo_state_policy(str(path), device="cpu")

        self.assertIsNotNone(loaded.gaussian_log_std)
        self.assertTrue(torch.allclose(loaded.gaussian_log_std.exp().cpu(), torch.tensor([0.11, 0.22, 0.33])))

    def test_bc_checkpoint_config_allows_runtime_flow_loss_overrides(self):
        from algos.rl.fpo_trainer import _merge_checkpoint_policy_config

        checkpoint_config = self.make_policy(
            sampling_steps=4,
            cfm_loss_huber_style="torch",
            cfm_loss_huber_delta=0.5,
        ).cfg.to_dict()
        params = Namespace(
            actor_hidden_dims=[999],
            critic_hidden_dims=[999],
            activation="relu",
            timestep_embed_dim=99,
            sampling_steps=8,
            actor_mlp_output_scale=0.75,
            actor_scale=1.0,
            action_clip=1.0,
            action_perturb_std=0.0,
            cfm_loss_t_inverse_cdf_beta=1.0,
            cfm_loss_reduction="sum",
            cfm_loss_use_huber=True,
            cfm_loss_huber_delta=1.0,
            cfm_loss_huber_style="fpo_control",
            flow_network_output_param="u",
            cfm_loss_mode="u",
        )

        merged = _merge_checkpoint_policy_config(
            checkpoint_config,
            params,
            obs_dim=6,
            action_dim=3,
        )

        self.assertEqual(merged.actor_hidden_dims, (32, 16))
        self.assertEqual(merged.critic_hidden_dims, (32, 16))
        self.assertEqual(merged.activation, "tanh")
        self.assertEqual(merged.timestep_embed_dim, 8)
        self.assertEqual(merged.sampling_steps, 8)
        self.assertEqual(merged.actor_mlp_output_scale, 0.75)
        self.assertEqual(merged.cfm_loss_huber_style, "fpo_control")
        self.assertEqual(merged.cfm_loss_huber_delta, 1.0)
        self.assertEqual(merged.cfm_loss_reduction, "sum")

    def test_policy_weight_loading_keeps_runtime_config_when_architecture_matches(self):
        from algos.rl.fpo_trainer import FPOStateTrainer, save_fpo_checkpoint

        checkpoint_policy = self.make_policy(sampling_steps=4, cfm_loss_huber_style="torch")
        runtime_policy = self.make_policy(sampling_steps=8, cfm_loss_huber_style="fpo_control")

        trainer = object.__new__(FPOStateTrainer)
        trainer.obs_dim = 6
        trainer.action_dim = 3
        trainer.device = torch.device("cpu")
        trainer.policy = runtime_policy
        trainer.current_stage = 0
        trainer.num_timesteps = 0
        trainer._make_optimizer = lambda: torch.optim.Adam(trainer.policy.parameters(), lr=1e-3)
        trainer.optimizer = trainer._make_optimizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bc.pt")
            save_fpo_checkpoint(
                path=path,
                policy=checkpoint_policy,
                optimizer_state_dict=None,
                num_timesteps=123,
            )
            trainer.load_policy_weights(path, load_optimizer=False)

        self.assertEqual(trainer.policy.cfg.sampling_steps, 8)
        self.assertEqual(trainer.policy.cfg.cfm_loss_huber_style, "fpo_control")
        self.assertEqual(trainer.policy.sampling_steps, 8)
        self.assertEqual(trainer.policy.cfm_loss_huber_style, "fpo_control")
        self.assertEqual(trainer.num_timesteps, 123)

    def test_curriculum_parity_advances_on_pregrasp_only_like_original_ppo(self):
        from algos.rl.fpo_trainer import should_advance_vividex_curriculum

        metrics = {
            "eval/mean_pregrasp_success": 1.0,
            "eval/mean_obj_lift": 0.0,
            "eval/mean_reward": 1.0,
        }

        self.assertTrue(
            should_advance_vividex_curriculum(
                current_stage=0,
                eval_metrics=metrics,
                pregrasp_threshold=0.95,
            )
        )
        self.assertFalse(
            should_advance_vividex_curriculum(
                current_stage=2,
                eval_metrics=metrics,
                pregrasp_threshold=0.95,
            )
        )

    def test_curriculum_stage1_metric_can_gate_stage0_advancement(self):
        from algos.rl.fpo_trainer import should_advance_vividex_curriculum

        metrics = {
            "eval/mean_pregrasp_success": 1.0,
            "eval/rollout_mean_stable_grasp_contact": 0.0,
        }

        self.assertFalse(
            should_advance_vividex_curriculum(
                current_stage=0,
                eval_metrics=metrics,
                pregrasp_threshold=0.95,
                stage1_metric="eval/rollout_mean_stable_grasp_contact",
                stage1_threshold=0.03,
            )
        )
        metrics["eval/rollout_mean_stable_grasp_contact"] = 0.05
        self.assertTrue(
            should_advance_vividex_curriculum(
                current_stage=0,
                eval_metrics=metrics,
                pregrasp_threshold=0.95,
                stage1_metric="eval/rollout_mean_stable_grasp_contact",
                stage1_threshold=0.03,
            )
        )

    def test_train_fpo_branch_uses_global_frequencies(self):
        train_py = Path(__file__).resolve().parents[1] / "tools" / "train.py"
        text = train_py.read_text()
        fpo_branch = text.split("elif cfg.agent.name == 'FPO':", 1)[1].split("else:", 1)[0]

        self.assertIn("eval_freq = int(cfg.eval_freq)", fpo_branch)
        self.assertIn("save_freq = int(cfg.save_freq)", fpo_branch)
        self.assertIn("restore_freq = int(cfg.restore_checkpoint_freq)", fpo_branch)
        self.assertNotIn("cfg.eval_freq // cfg.n_envs", fpo_branch)
        self.assertNotIn("cfg.save_freq // cfg.n_envs", fpo_branch)
        self.assertNotIn("cfg.restore_checkpoint_freq // cfg.n_envs", fpo_branch)

    def test_train_resume_resolution_keeps_absolute_external_checkpoint(self):
        import ast
        import tempfile

        train_py = Path(__file__).resolve().parents[1] / "tools" / "train.py"
        module = ast.parse(train_py.read_text())
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_resume_model")
        namespace = {"os": os}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(train_py), "exec"), namespace)
        resolve_resume_model = namespace["resolve_resume_model"]

        with tempfile.TemporaryDirectory() as tmpdir:
            external = os.path.join(tmpdir, "models", "best.pt")
            os.makedirs(os.path.dirname(external), exist_ok=True)
            Path(external).write_bytes(b"checkpoint")

            resolved = resolve_resume_model(tmpdir, external, "FPO")

        self.assertEqual(resolved, external)

    def test_train_resume_resolution_prefers_local_restore_checkpoint(self):
        import ast
        import tempfile

        train_py = Path(__file__).resolve().parents[1] / "tools" / "train.py"
        module = ast.parse(train_py.read_text())
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_resume_model")
        namespace = {"os": os}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(train_py), "exec"), namespace)
        resolve_resume_model = namespace["resolve_resume_model"]

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "restore_checkpoint.pt").write_bytes(b"checkpoint")

            resolved = resolve_resume_model(tmpdir, None, "FPO")

        self.assertEqual(resolved, "restore_checkpoint.pt")

    def test_collect_bc_resolves_explicit_checkpoint_file_and_config_dir(self):
        from tools.collect_fpo_bc_dataset import _resolve_checkpoint_inputs

        args = Namespace(
            checkpoint_dir=None,
            checkpoint_file="/runs/ppo/logs/rl_models_10000000_steps.zip",
            config_dir="/runs/ppo",
        )

        checkpoint_path, config_path = _resolve_checkpoint_inputs(args)

        self.assertEqual(checkpoint_path, "/runs/ppo/logs/rl_models_10000000_steps.zip")
        self.assertEqual(config_path, "/runs/ppo/exp_config.yaml")

    def test_collect_bc_keeps_restore_checkpoint_dir_compatibility(self):
        from tools.collect_fpo_bc_dataset import _resolve_checkpoint_inputs

        args = Namespace(
            checkpoint_dir="/runs/ppo",
            checkpoint_file=None,
            config_dir=None,
        )

        checkpoint_path, config_path = _resolve_checkpoint_inputs(args)

        self.assertEqual(checkpoint_path, "/runs/ppo/restore_checkpoint.zip")
        self.assertEqual(config_path, "/runs/ppo/exp_config.yaml")

    def test_fpo_config_exposes_optional_bc_anchor_defaults(self):
        config_path = Path(__file__).resolve().parents[1] / "algos" / "rl" / "config" / "agent" / "fpo.yaml"
        text = config_path.read_text()

        self.assertIn("bc_anchor_dataset: null", text)
        self.assertIn("bc_anchor_coef: 0.0", text)
        self.assertIn("bc_anchor_min_coef: 0.0", text)
        self.assertIn("bc_anchor_decay_steps: 0", text)
        self.assertIn("cfm_loss_huber_style: torch", text)
        self.assertIn("log_ratio_scale: 1.0", text)
        self.assertIn("cfm_diff_clip_min: null", text)
        self.assertIn("cfm_diff_clip_max: null", text)
        self.assertIn("cfm_loss_clamp_negative_advantages: null", text)
        self.assertIn("actor_objective: fpo", text)
        self.assertIn("gaussian_action_std: 0.05", text)
        self.assertIn("gaussian_mean_source: policy", text)
        self.assertIn("gaussian_action_std_source: fixed", text)
        self.assertIn("gaussian_action_std_scale: 1.0", text)
        self.assertIn("gaussian_action_std_min_scale: 0.0", text)
        self.assertIn("gaussian_action_std_decay_steps: 0", text)
        self.assertIn("gaussian_action_std_trainable: False", text)
        self.assertIn("ent_coef: 0.0", text)
        self.assertIn("fpo_chunk_steps: 1", text)
        self.assertIn("average_cfm_loss_in_chunk: False", text)
        self.assertIn("rollout_deterministic: False", text)
        self.assertIn("rollout_action_noise_std: 0.0", text)
        self.assertIn("advantage_clamp_positive: null", text)
        self.assertIn("advantage_clamp_negative: null", text)
        self.assertIn("action_anchor_coef: 0.0", text)
        self.assertIn("on_policy_action_anchor_coef: 0.0", text)
        self.assertIn("obj_precision_penalty_coef: 0.0", text)
        self.assertIn("obj_precision_penalty_target: 0.001", text)
        self.assertIn("obj_precision_penalty_power: 1.0", text)
        self.assertIn("obj_precision_penalty_max: 1.0", text)
        self.assertIn("obj_precision_delta_coef: 0.0", text)
        self.assertIn("obj_precision_delta_target: 0.0011", text)
        self.assertIn("obj_precision_delta_max: 1.0", text)
        self.assertIn("ppo_base_checkpoint: null", text)
        self.assertIn("flow_residual_coef: 1.0", text)
        self.assertIn("trainable_ppo_base_actor: False", text)
        self.assertIn("ppo_base_actor_lr_scale: 0.25", text)
        self.assertIn("ppo_teacher_checkpoint: null", text)
        self.assertIn("ppo_teacher_action_anchor_coef: 0.0", text)
        self.assertIn("ppo_teacher_action_anchor_min_coef: 0.0", text)
        self.assertIn("advantage_weighted_action_coef: 0.0", text)
        self.assertIn("advantage_weighted_action_positive_only: True", text)
        self.assertIn("advantage_weighted_action_mode: action", text)
        self.assertIn("precision_elite_action_coef: 0.0", text)
        self.assertIn("precision_elite_err_threshold: 0.003", text)
        self.assertIn("precision_elite_min_fraction: 0.0", text)
        self.assertIn("precision_elite_mode: action", text)
        self.assertIn("actor_max_grad_norm: 1.0", text)
        self.assertIn("critic_max_grad_norm: 1.0", text)
        self.assertIn("flow_actor_lr_scale: 1.0", text)
        self.assertIn("residual_head_lr_scale: 1.0", text)
        self.assertIn("direct_head_lr_scale: 1.0", text)
        self.assertIn("gaussian_log_std_lr_scale: 1.0", text)
        self.assertIn("residual_head_max_grad_norm: 1.0", text)
        self.assertIn("direct_head_max_grad_norm: 1.0", text)
        self.assertIn("positive_advantage_only: False", text)
        self.assertIn("train_actor_after_iterations: 0", text)
        self.assertIn("save_best_model: True", text)
        self.assertIn("best_metric: eval/mean_reward", text)
        self.assertIn("rfpo_precision_score_stage_min: 2.0", text)
        self.assertIn("rfpo_precision_score_pregrasp_min: 0.98", text)
        self.assertIn("rfpo_precision_score_obj_err_target: 0.0011", text)
        self.assertIn("rfpo_precision_score_obj_err_coef: 500.0", text)
        self.assertIn("rfpo_precision_score_obj_err_cap: 2.0", text)

    def test_eval_safety_can_early_stop_on_same_eval_that_rolls_back(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            (models_dir / "best.pt").write_bytes(b"checkpoint")

            loaded = []
            trainer.output_dir = tmpdir
            trainer.best_metric = "eval/mean_reward"
            trainer.best_metric_mode = "max"
            trainer.save_best_model = True
            trainer.rollback_to_best_on_degrade = True
            trainer.rollback_patience = 2
            trainer.reset_optimizer_on_rollback = False
            trainer.early_stop_on_degrade = True
            trainer.early_stop_patience = 2
            trainer.degrade_metric = "eval/mean_reward"
            trainer.degrade_threshold = 1.0
            trainer.best_metric_value = 70.0
            trainer.best_num_timesteps = 100
            trainer.degrade_count = 1
            trainer.num_timesteps = 200
            trainer.iterations = 3
            trainer.stop_training = False
            trainer.load_policy_weights = lambda path, load_optimizer: loaded.append((path, load_optimizer))

            metrics = trainer._handle_eval_safety({"eval/mean_reward": 68.0})

        self.assertEqual(len(loaded), 1)
        self.assertEqual(metrics["train/rolled_back_to_best"], 1.0)
        self.assertEqual(metrics["train/early_stop_triggered"], 1.0)
        self.assertTrue(trainer.stop_training)

    def test_bc_anchor_coef_decays_linearly_when_enabled(self):
        from algos.rl.fpo_trainer import linear_bc_anchor_coef

        self.assertEqual(linear_bc_anchor_coef(0.1, 0, 0), 0.1)
        self.assertEqual(linear_bc_anchor_coef(0.1, 100, 0), 0.1)
        self.assertAlmostEqual(linear_bc_anchor_coef(0.1, 100, 50), 0.05)
        self.assertAlmostEqual(linear_bc_anchor_coef(0.1, 100, 100), 0.0)
        self.assertAlmostEqual(linear_bc_anchor_coef(0.1, 100, 200), 0.0)

    def test_bc_anchor_coef_decays_to_configured_floor(self):
        from algos.rl.fpo_trainer import linear_bc_anchor_coef

        self.assertAlmostEqual(linear_bc_anchor_coef(0.1, 100, 50, min_coef=0.02), 0.06)
        self.assertAlmostEqual(linear_bc_anchor_coef(0.1, 100, 100, min_coef=0.02), 0.02)
        self.assertAlmostEqual(linear_bc_anchor_coef(0.1, 100, 200, min_coef=0.02), 0.02)

    def test_action_anchor_loss_matches_requested_loss_type(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.action_anchor_loss = "mse"
        trainer.action_anchor_huber_delta = 0.05
        pred = torch.tensor([[0.0, 0.5, 1.0]])
        target = torch.tensor([[0.0, 0.0, 1.0]])
        self.assertAlmostEqual(float(trainer._action_loss(pred, target)), 0.25 / 3.0)

        trainer.action_anchor_loss = "huber"
        self.assertGreater(float(trainer._action_loss(pred, target)), 0.0)

    def test_critic_distillation_reduces_teacher_value_mse(self):
        from tools.distill_fpo_critic_from_ppo import distill_critic

        torch.manual_seed(0)
        policy = self.make_policy()
        observations = torch.randn(128, 6)
        weights = torch.tensor([[0.5], [-0.25], [0.1], [0.3], [-0.4], [0.2]])
        target_values = observations @ weights + 0.1

        stats = distill_critic(
            policy=policy,
            observations=observations.numpy().astype("float32"),
            target_values=target_values.numpy().astype("float32"),
            steps=120,
            batch_size=32,
            learning_rate=1e-2,
            weight_decay=0.0,
            max_grad_norm=10.0,
            device=torch.device("cpu"),
            log_freq=0,
        )

        self.assertLess(stats["final_mse"], stats["initial_mse"] * 0.25)

    def test_best_metric_logic_supports_max_and_min_modes(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.best_metric_mode = "max"
        self.assertTrue(trainer._is_better_metric(2.0, None))
        self.assertTrue(trainer._is_better_metric(2.0, 1.0))
        self.assertFalse(trainer._is_better_metric(1.0, 2.0))

        trainer.best_metric_mode = "min"
        self.assertTrue(trainer._is_better_metric(1.0, 2.0))
        self.assertFalse(trainer._is_better_metric(2.0, 1.0))

    def test_rfpo_precision_score_rewards_close_object_error_after_stage_is_solved(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.rfpo_precision_score_stage_min = 2.0
        trainer.rfpo_precision_score_pregrasp_min = 0.98
        trainer.rfpo_precision_score_obj_err_target = 0.0011
        trainer.rfpo_precision_score_obj_err_coef = 500.0
        trainer.rfpo_precision_score_obj_err_cap = 2.0

        high_reward_bad_precision = trainer._rfpo_precision_score(
            {
                "eval/mean_reward": 75.6,
                "eval/mean_stage": 2.0,
                "eval/mean_pregrasp_success": 1.0,
                "eval/mean_obj_com_err": 0.00436,
            }
        )
        close_reward_good_precision = trainer._rfpo_precision_score(
            {
                "eval/mean_reward": 75.2,
                "eval/mean_stage": 2.0,
                "eval/mean_pregrasp_success": 1.0,
                "eval/mean_obj_com_err": 0.00292,
            }
        )
        unsolved_stage = trainer._rfpo_precision_score(
            {
                "eval/mean_reward": 76.0,
                "eval/mean_stage": 1.0,
                "eval/mean_pregrasp_success": 1.0,
                "eval/mean_obj_com_err": 0.001,
            }
        )

        self.assertGreater(close_reward_good_precision, high_reward_bad_precision)
        self.assertLess(unsolved_stage, high_reward_bad_precision)

    def test_eval_safety_can_save_best_by_rfpo_precision_score(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.best_metric = "eval/rfpo_precision_score"
        trainer.degrade_metric = "eval/rfpo_precision_score"
        trainer.best_metric_mode = "max"
        trainer.best_metric_value = None
        trainer.best_num_timesteps = 0
        trainer.degrade_count = 0
        trainer.degrade_threshold = 1.0
        trainer.rollback_to_best_on_degrade = False
        trainer.rollback_patience = 0
        trainer.reset_optimizer_on_rollback = False
        trainer.early_stop_on_degrade = False
        trainer.early_stop_patience = 0
        trainer.save_best_model = False
        trainer.stop_training = False
        trainer.num_timesteps = 200
        trainer.rfpo_precision_score_stage_min = 2.0
        trainer.rfpo_precision_score_pregrasp_min = 0.98
        trainer.rfpo_precision_score_obj_err_target = 0.0011
        trainer.rfpo_precision_score_obj_err_coef = 500.0
        trainer.rfpo_precision_score_obj_err_cap = 2.0

        metrics = trainer._handle_eval_safety(
            {
                "eval/mean_reward": 75.2,
                "eval/mean_stage": 2.0,
                "eval/mean_pregrasp_success": 1.0,
                "eval/mean_obj_com_err": 0.00292,
            }
        )

        self.assertIn("eval/rfpo_precision_score", metrics)
        self.assertEqual(trainer.best_metric_value, metrics["eval/rfpo_precision_score"])
        self.assertEqual(trainer.best_num_timesteps, 200)

    def test_eval_rollback_preserves_training_step_counter(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.best_metric = "eval/mean_reward"
        trainer.degrade_metric = "eval/mean_reward"
        trainer.best_metric_mode = "max"
        trainer.best_metric_value = 70.0
        trainer.best_num_timesteps = 100
        trainer.degrade_count = 0
        trainer.degrade_threshold = 1.0
        trainer.rollback_to_best_on_degrade = True
        trainer.rollback_patience = 1
        trainer.reset_optimizer_on_rollback = False
        trainer.early_stop_on_degrade = False
        trainer.early_stop_patience = 0
        trainer.save_best_model = False
        trainer.output_dir = "/tmp/does-not-matter"
        trainer.stop_training = False
        trainer.num_timesteps = 200
        trainer.iterations = 3
        trainer._is_better_metric = lambda value, reference: False

        calls = []
        trainer.load_policy_weights = lambda path, load_optimizer: calls.append((path, load_optimizer))
        old_exists = os.path.exists
        os.path.exists = lambda path: True
        try:
            metrics = trainer._handle_eval_safety({"eval/mean_reward": 50.0})
        finally:
            os.path.exists = old_exists

        self.assertEqual(trainer.num_timesteps, 200)
        self.assertEqual(trainer.iterations, 3)
        self.assertEqual(metrics["train/rolled_back_to_best"], 1.0)
        self.assertEqual(len(calls), 1)

    def test_advantage_weighted_action_loss_ignores_nonpositive_advantages(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.advantage_weighted_action_positive_only = True
        trainer.advantage_weighted_action_temp = 1.0
        trainer.advantage_weighted_action_max_weight = 20.0
        trainer.advantage_weighted_action_mode = "action"
        trainer.action_anchor_huber_delta = 0.05
        trainer.policy = self.make_policy()
        batch = Namespace(
            obs=torch.randn(4, 6),
            actions=torch.randn(4, 3).clamp(-1.0, 1.0),
            advantages=torch.tensor([[-1.0], [-0.1], [0.0], [-2.0]]),
        )

        self.assertIsNone(trainer._advantage_weighted_action_loss(batch, coef=1.0))

        batch.advantages = torch.tensor([[-1.0], [0.2], [0.0], [1.0]])
        loss = trainer._advantage_weighted_action_loss(batch, coef=1.0)
        self.assertIsNotNone(loss)
        self.assertGreaterEqual(float(loss.detach()), 0.0)

    def test_advantage_weighted_flow_loss_uses_positive_advantage_actions_as_cfm_targets(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        class FlowPolicy(torch.nn.Module):
            cfm_loss_t_inverse_cdf_beta = 1.5

            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.tensor(1.0))
                self.last_obs_shape = None
                self.last_action_shape = None

            def get_cfm_loss(self, obs, actions, eps, t):
                self.last_obs_shape = tuple(obs.shape)
                self.last_action_shape = tuple(actions.shape)
                return (actions.square().mean(dim=-1, keepdim=True) * self.param).expand(-1, eps.shape[1]), None, None

        trainer = object.__new__(FPOStateTrainer)
        trainer.advantage_weighted_action_positive_only = True
        trainer.advantage_weighted_action_temp = 1.0
        trainer.advantage_weighted_action_max_weight = 20.0
        trainer.advantage_weighted_action_mode = "flow"
        trainer.action_anchor_huber_delta = 0.05
        trainer.action_dim = 3
        trainer.device = torch.device("cpu")
        trainer.n_cfm_samples = 2
        trainer.policy = FlowPolicy()
        batch = Namespace(
            obs=torch.randn(5, 6),
            actions=torch.randn(5, 3).clamp(-1.0, 1.0),
            advantages=torch.tensor([[-1.0], [0.0], [0.2], [1.0], [-0.5]]),
        )

        loss = trainer._advantage_weighted_action_loss(batch, coef=1.0)

        self.assertIsNotNone(loss)
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertEqual(trainer.policy.last_obs_shape, (2, 6))
        self.assertEqual(trainer.policy.last_action_shape, (2, 3))

    def test_rollout_buffer_keeps_object_precision_metadata(self):
        from algos.rl.fpo_rollout_buffer import FPOTransition, FPORolloutBuffer

        buffer = FPORolloutBuffer(
            num_envs=2,
            num_steps=2,
            obs_dim=6,
            action_dim=3,
            n_cfm_samples=2,
            device="cpu",
        )
        for step in range(2):
            buffer.add(
                FPOTransition(
                    obs=torch.zeros(2, 6),
                    actions=torch.zeros(2, 3),
                    flow_actions=torch.zeros(2, 3),
                    rewards=torch.zeros(2),
                    dones=torch.zeros(2),
                    values=torch.zeros(2, 1),
                    old_cfm_loss=torch.zeros(2, 2),
                    old_x1_pred=torch.zeros(2, 2, 3),
                    cfm_eps=torch.zeros(2, 2, 3),
                    cfm_t=torch.zeros(2, 2, 1),
                    old_log_prob=torch.zeros(2, 1),
                    obj_com_err=torch.tensor([0.001 + step, 0.004 + step]),
                    hand_mjpos_err=torch.tensor([0.02 + step, 0.03 + step]),
                    control_error=torch.tensor([0.001 + step, 0.002 + step]),
                    stage=torch.tensor([2.0, 1.0]),
                )
            )

        buffer.compute_returns_and_advantages(
            last_values=torch.zeros(2, 1),
            gamma=0.95,
            gae_lambda=0.95,
            normalize_advantage=False,
        )
        batch = next(buffer.iter_minibatches(num_minibatches=1, num_epochs=1))

        self.assertEqual(tuple(batch.obj_com_err.shape), (4, 1))
        self.assertEqual(tuple(batch.hand_mjpos_err.shape), (4, 1))
        self.assertEqual(tuple(batch.control_error.shape), (4, 1))
        self.assertEqual(tuple(batch.env_rewards.shape), (4, 1))
        self.assertEqual(tuple(batch.stage.shape), (4, 1))
        self.assertTrue(torch.isfinite(batch.obj_com_err).all())

    def test_precision_elite_action_loss_uses_low_error_stage_two_samples(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.precision_elite_stage_min = 2.0
        trainer.precision_elite_err_threshold = 0.003
        trainer.precision_elite_max_weight = 10.0
        trainer.precision_elite_temp = 0.001
        trainer.precision_elite_mode = "action"
        trainer.precision_elite_top_fraction = 1.0
        trainer.precision_elite_reward_quantile = 0.0
        trainer.precision_elite_hand_threshold = None
        trainer.precision_elite_control_threshold = None
        trainer.action_anchor_loss = "huber"
        trainer.action_anchor_huber_delta = 0.05
        trainer.device = torch.device("cpu")
        trainer.obs_dim = 6
        trainer.action_dim = 3
        trainer.n_cfm_samples = 2
        trainer.policy = self.make_policy()
        batch = Namespace(
            obs=torch.randn(4, 6),
            actions=torch.randn(4, 3).clamp(-1.0, 1.0),
            obj_com_err=torch.tensor([[0.002], [0.006], [0.0015], [0.002]]),
            stage=torch.tensor([[2.0], [2.0], [1.0], [2.0]]),
        )

        loss = trainer._precision_elite_action_loss(batch, coef=1.0)

        self.assertIsNotNone(loss)
        self.assertGreaterEqual(float(loss.detach()), 0.0)

        batch.obj_com_err = torch.full((4, 1), 0.02)
        self.assertIsNone(trainer._precision_elite_action_loss(batch, coef=1.0))

    def test_precision_elite_gaussian_nll_uses_log_prob_of_elite_actions(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.precision_elite_stage_min = 2.0
        trainer.precision_elite_err_threshold = 0.003
        trainer.precision_elite_max_weight = 10.0
        trainer.precision_elite_temp = 0.001
        trainer.precision_elite_mode = "gaussian_nll"
        trainer.precision_elite_top_fraction = 1.0
        trainer.precision_elite_reward_quantile = 0.0
        trainer.precision_elite_hand_threshold = None
        trainer.precision_elite_control_threshold = None
        trainer.gaussian_mean_source = "policy"
        trainer.device = torch.device("cpu")
        trainer.obs_dim = 6
        trainer.action_dim = 3
        trainer.n_cfm_samples = 2
        trainer.gaussian_action_std_source = "fixed"
        trainer.gaussian_action_std = 0.05
        trainer.gaussian_log_std = None
        trainer.gaussian_action_std_scale = 1.0
        trainer.gaussian_action_std_min_scale = 1.0
        trainer.gaussian_action_std_decay_steps = 0
        trainer.num_timesteps = 0
        trainer.policy = self.make_policy()
        batch = Namespace(
            obs=torch.zeros(4, 6),
            actions=torch.full((4, 3), 0.4),
            obj_com_err=torch.tensor([[0.002], [0.006], [0.0015], [0.002]]),
            stage=torch.tensor([[2.0], [2.0], [1.0], [2.0]]),
        )

        loss = trainer._precision_elite_action_loss(batch, coef=1.0)

        self.assertIsNotNone(loss)
        self.assertGreater(float(loss.detach()), 1.0)

    def test_precision_elite_can_select_top_joint_quality_samples(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.precision_elite_stage_min = 2.0
        trainer.precision_elite_err_threshold = 0.006
        trainer.precision_elite_max_weight = 10.0
        trainer.precision_elite_temp = 0.001
        trainer.precision_elite_mode = "action"
        trainer.precision_elite_top_fraction = 0.5
        trainer.precision_elite_reward_quantile = 0.5
        trainer.precision_elite_hand_threshold = 0.026
        trainer.precision_elite_control_threshold = 0.01
        trainer.action_anchor_loss = "huber"
        trainer.action_anchor_huber_delta = 0.05
        trainer.device = torch.device("cpu")
        trainer.obs_dim = 6
        trainer.action_dim = 3
        trainer.n_cfm_samples = 2
        trainer.policy = self.make_policy()
        batch = Namespace(
            obs=torch.randn(6, 6),
            actions=torch.randn(6, 3).clamp(-1.0, 1.0),
            obj_com_err=torch.tensor([[0.0012], [0.0020], [0.0030], [0.0040], [0.0050], [0.0010]]),
            hand_mjpos_err=torch.tensor([[0.020], [0.024], [0.030], [0.021], [0.022], [0.028]]),
            control_error=torch.tensor([[0.002], [0.003], [0.002], [0.020], [0.002], [0.002]]),
            env_rewards=torch.tensor([[0.9], [0.8], [0.7], [1.0], [0.6], [1.1]]),
            stage=torch.full((6, 1), 2.0),
        )

        loss = trainer._precision_elite_action_loss(batch, coef=1.0)

        self.assertIsNotNone(loss)
        stats = trainer._last_precision_elite_stats
        self.assertLess(stats["frac"], 0.5)
        self.assertLessEqual(stats["err_mean"], 0.002)
        self.assertLessEqual(stats["hand_mean"], 0.026)
        self.assertGreaterEqual(stats["reward_mean"], 0.8)

    def test_precision_elite_min_fraction_falls_back_to_best_stage_two_samples(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.precision_elite_stage_min = 2.0
        trainer.precision_elite_err_threshold = 0.001
        trainer.precision_elite_min_fraction = 0.5
        trainer.precision_elite_max_weight = 10.0
        trainer.precision_elite_temp = 0.001
        trainer.precision_elite_mode = "action"
        trainer.precision_elite_top_fraction = 1.0
        trainer.precision_elite_reward_quantile = 0.0
        trainer.precision_elite_hand_threshold = None
        trainer.precision_elite_control_threshold = None
        trainer.action_anchor_loss = "huber"
        trainer.action_anchor_huber_delta = 0.05
        trainer.device = torch.device("cpu")
        trainer.obs_dim = 6
        trainer.action_dim = 3
        trainer.n_cfm_samples = 2
        trainer.policy = self.make_policy()
        batch = Namespace(
            obs=torch.randn(6, 6),
            actions=torch.randn(6, 3).clamp(-1.0, 1.0),
            obj_com_err=torch.tensor([[0.0060], [0.0020], [0.0040], [0.0030], [0.0050], [0.0025]]),
            stage=torch.full((6, 1), 2.0),
        )

        loss = trainer._precision_elite_action_loss(batch, coef=1.0)

        self.assertIsNotNone(loss)
        stats = trainer._last_precision_elite_stats
        self.assertAlmostEqual(stats["frac"], 0.5)
        self.assertLessEqual(stats["err_mean"], 0.0026)

    def test_precision_elite_min_fraction_survives_top_fraction_filter(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.precision_elite_stage_min = 2.0
        trainer.precision_elite_err_threshold = 0.001
        trainer.precision_elite_min_fraction = 0.5
        trainer.precision_elite_max_weight = 10.0
        trainer.precision_elite_temp = 0.001
        trainer.precision_elite_mode = "action"
        trainer.precision_elite_top_fraction = 0.1
        trainer.precision_elite_reward_quantile = 0.0
        trainer.precision_elite_hand_threshold = None
        trainer.precision_elite_control_threshold = None
        trainer.action_anchor_huber_delta = 0.05
        trainer.device = torch.device("cpu")
        trainer.obs_dim = 6
        trainer.action_dim = 3
        trainer.n_cfm_samples = 2
        trainer.policy = self.make_policy()
        batch = Namespace(
            obs=torch.randn(6, 6),
            actions=torch.randn(6, 3).clamp(-1.0, 1.0),
            obj_com_err=torch.tensor([[0.0060], [0.0020], [0.0040], [0.0030], [0.0050], [0.0025]]),
            stage=torch.full((6, 1), 2.0),
        )

        loss = trainer._precision_elite_action_loss(batch, coef=1.0)

        self.assertIsNotNone(loss)
        stats = trainer._last_precision_elite_stats
        self.assertAlmostEqual(stats["frac"], 0.5)
        self.assertLessEqual(stats["err_mean"], 0.0032)

    def test_self_elite_replay_keeps_best_low_error_actions(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.obs_dim = 6
        trainer.action_dim = 3
        trainer.self_elite_replay_capacity = 3
        trainer.self_elite_replay_err_threshold = 0.003
        trainer.self_elite_replay_stage_min = 2.0
        trainer.self_elite_replay_hand_threshold = None
        trainer.self_elite_replay_control_threshold = None
        trainer.self_elite_replay_min_reward = None
        trainer.self_elite_replay_obs = None
        trainer.self_elite_replay_actions = None
        trainer.self_elite_replay_obj_err = None
        trainer.self_elite_replay_rewards = None
        batch = Namespace(
            obs=torch.arange(30, dtype=torch.float32).reshape(5, 6),
            actions=torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10.0,
            obj_com_err=torch.tensor([[0.0040], [0.0010], [0.0025], [0.0008], [0.0035]]),
            stage=torch.tensor([[2.0], [2.0], [1.0], [2.0], [2.0]]),
            env_rewards=torch.tensor([[0.1], [0.8], [0.7], [0.9], [0.2]]),
        )

        added = trainer._update_self_elite_replay(batch)

        self.assertEqual(added, 2)
        self.assertEqual(tuple(trainer.self_elite_replay_obs.shape), (2, 6))
        self.assertTrue(torch.allclose(trainer.self_elite_replay_obj_err, torch.tensor([0.0008, 0.0010])))

        batch.obj_com_err = torch.tensor([[0.0020], [0.0006], [0.0028], [0.0018], [0.0022]])
        batch.stage = torch.full((5, 1), 2.0)
        added = trainer._update_self_elite_replay(batch)

        self.assertEqual(added, 5)
        self.assertEqual(tuple(trainer.self_elite_replay_obs.shape), (3, 6))
        self.assertTrue(torch.all(trainer.self_elite_replay_obj_err[:-1] <= trainer.self_elite_replay_obj_err[1:]))
        self.assertLessEqual(float(trainer.self_elite_replay_obj_err[-1]), 0.0018)

    def test_self_elite_replay_loss_replays_stored_actions(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.self_elite_replay_coef = 1.0
        trainer.self_elite_replay_capacity = 4
        trainer.self_elite_replay_batch_size = 3
        trainer.self_elite_replay_mode = "action"
        trainer.self_elite_replay_weight_temp = 0.001
        trainer.self_elite_replay_max_weight = 5.0
        trainer.action_anchor_loss = "huber"
        trainer.action_anchor_huber_delta = 0.05
        trainer.device = torch.device("cpu")
        trainer.policy = self.make_policy()
        trainer.self_elite_replay_obs = torch.randn(4, 6)
        trainer.self_elite_replay_actions = torch.randn(4, 3).clamp(-1.0, 1.0)
        trainer.self_elite_replay_obj_err = torch.tensor([0.0010, 0.0020, 0.0008, 0.0015])
        trainer.self_elite_replay_rewards = torch.tensor([0.8, 0.5, 0.9, 0.7])

        loss = trainer._self_elite_replay_loss(coef=1.0)

        self.assertIsNotNone(loss)
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertGreater(trainer._last_self_elite_replay_stats["size"], 0)

    def test_precision_advantage_adds_bonus_to_low_error_stage_two_samples(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.advantage_clamp_negative = None
        trainer.advantage_clamp_positive = None
        trainer.precision_advantage_coef = 2.0
        trainer.precision_advantage_target = 0.001
        trainer.precision_advantage_threshold = 0.003
        trainer.precision_advantage_stage_min = 2.0
        trainer.precision_advantage_power = 1.0
        trainer.precision_advantage_max_bonus = 3.0
        trainer.precision_advantage_top_fraction = 1.0
        trainer.precision_advantage_hand_threshold = None
        trainer.precision_advantage_control_threshold = None
        advantages = torch.zeros(4, 1)
        batch = Namespace(
            obj_com_err=torch.tensor([[0.0010], [0.0020], [0.0040], [0.0015]]),
            stage=torch.tensor([[2.0], [2.0], [2.0], [1.0]]),
        )

        shaped = trainer._precision_shaped_advantages(batch, advantages)

        expected = torch.tensor([[2.0], [1.0], [0.0], [0.0]])
        self.assertTrue(torch.allclose(shaped, expected, atol=1e-6))
        self.assertAlmostEqual(trainer._last_precision_advantage_stats["frac"], 0.5)
        self.assertAlmostEqual(trainer._last_precision_advantage_stats["bonus_mean"], 1.5, places=6)

    def test_precision_advantage_respects_filters_and_top_fraction(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.advantage_clamp_negative = None
        trainer.advantage_clamp_positive = None
        trainer.precision_advantage_coef = 1.0
        trainer.precision_advantage_target = 0.001
        trainer.precision_advantage_threshold = 0.004
        trainer.precision_advantage_stage_min = 2.0
        trainer.precision_advantage_power = 1.0
        trainer.precision_advantage_max_bonus = 2.0
        trainer.precision_advantage_top_fraction = 0.5
        trainer.precision_advantage_hand_threshold = 0.02
        trainer.precision_advantage_control_threshold = 0.006
        advantages = torch.zeros(5, 1)
        batch = Namespace(
            obj_com_err=torch.tensor([[0.0030], [0.0015], [0.0020], [0.0012], [0.0035]]),
            stage=torch.full((5, 1), 2.0),
            hand_mjpos_err=torch.tensor([[0.010], [0.030], [0.010], [0.010], [0.010]]),
            control_error=torch.tensor([[0.004], [0.004], [0.008], [0.004], [0.004]]),
        )

        shaped = trainer._precision_shaped_advantages(batch, advantages)

        self.assertGreater(float(shaped[3]), 0.0)
        self.assertEqual(float(shaped[0]), 0.0)
        self.assertEqual(float(shaped[1]), 0.0)
        self.assertEqual(float(shaped[2]), 0.0)
        self.assertEqual(float(shaped[4]), 0.0)
        self.assertAlmostEqual(trainer._last_precision_advantage_stats["frac"], 0.2)

    def test_ppo_teacher_action_anchor_loss_uses_teacher_predictions(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        class Teacher:
            def predict(self, obs, deterministic=True):
                self.last_deterministic = deterministic
                return torch.ones(obs.shape[0], 3).numpy() * 0.25, None

        trainer = object.__new__(FPOStateTrainer)
        trainer.ppo_teacher = Teacher()
        trainer.ppo_teacher_action_anchor_deterministic = True
        trainer.action_anchor_loss = "mse"
        trainer.action_anchor_huber_delta = 0.05
        trainer.device = torch.device("cpu")
        trainer.policy = self.make_policy()
        batch = Namespace(obs=torch.randn(5, 6))

        loss = trainer._ppo_teacher_action_anchor_loss(batch, coef=1.0)

        self.assertIsNotNone(loss)
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertTrue(trainer.ppo_teacher.last_deterministic)

    def test_fpo_trainer_predict_samples_gaussian_action_when_requested(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        class Policy:
            training = True

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

            def sample_gaussian_action(self, obs, action_std):
                self.sample_called = True
                return torch.ones(obs.shape[0], 3) * 0.5, None, None

            def act(self, obs, deterministic=True):
                self.act_called = True
                return torch.zeros(obs.shape[0], 3)

        trainer = object.__new__(FPOStateTrainer)
        trainer.device = torch.device("cpu")
        trainer.actor_objective = "gaussian_ppo"
        trainer.policy = Policy()
        trainer._current_action_std = lambda: 0.05

        actions, _ = trainer.predict(np.zeros((2, 6), dtype=np.float32), deterministic=False)

        self.assertTrue(trainer.policy.sample_called)
        self.assertFalse(hasattr(trainer.policy, "act_called"))
        self.assertTrue(trainer.policy.training)
        self.assertTrue(np.allclose(actions, 0.5))

    def test_loaded_fpo_policy_samples_gaussian_action_when_requested(self):
        from algos.rl.fpo_trainer import LoadedFPOPolicy

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(()))

            def sample_gaussian_action(self, obs, action_std):
                self.sample_called = True
                return torch.ones(obs.shape[0], 3) * 0.25, None, None

            def act(self, obs, deterministic=True):
                self.act_called = True
                return torch.zeros(obs.shape[0], 3)

        policy = Policy()
        loaded = LoadedFPOPolicy(
            policy=policy,
            num_timesteps=10,
            extra={"agent_params": {"actor_objective": "gaussian_ppo", "gaussian_action_std": 0.05}},
        )

        actions, _ = loaded.predict(np.zeros((2, 6), dtype=np.float32), deterministic=False)

        self.assertTrue(policy.sample_called)
        self.assertFalse(hasattr(policy, "act_called"))
        self.assertTrue(np.allclose(actions, 0.25))

    def test_hybrid_actor_objective_uses_gaussian_exploration_and_mixes_fpo_weight(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        class Policy:
            training = True

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

            def sample_gaussian_action(self, obs, action_std):
                self.sample_called = True
                return torch.ones(obs.shape[0], 3) * 0.75, None, None

            def act(self, obs, deterministic=True):
                self.act_called = True
                return torch.zeros(obs.shape[0], 3)

        trainer = object.__new__(FPOStateTrainer)
        trainer.device = torch.device("cpu")
        trainer.actor_objective = "hybrid_fpo"
        trainer.policy = Policy()
        trainer._current_action_std = lambda: 0.05
        trainer.fpo_objective_coef = 1.0
        trainer.fpo_objective_min_coef = 0.0
        trainer.fpo_objective_decay_steps = 100
        trainer.gaussian_objective_coef = 1.0
        trainer.gaussian_objective_min_coef = 0.0
        trainer.gaussian_objective_decay_steps = 100
        trainer.num_timesteps = 50

        actions, _ = trainer.predict(np.zeros((2, 6), dtype=np.float32), deterministic=False)

        self.assertTrue(trainer.policy.sample_called)
        self.assertFalse(hasattr(trainer.policy, "act_called"))
        self.assertTrue(trainer.policy.training)
        self.assertTrue(np.allclose(actions, 0.75))
        weights = trainer._objective_weights()
        self.assertAlmostEqual(weights["fpo"], 0.5)
        self.assertAlmostEqual(weights["gaussian"], 0.5)

    def test_fpo_eval_callback_defaults_to_stochastic_evaluation(self):
        from hand_imitation.utils.fpo_eval import FPOEvalCallback

        callback = FPOEvalCallback("/tmp", n_eval_episodes=3)
        seen = {}

        def fake_evaluate_policy(*args, **kwargs):
            seen["deterministic"] = kwargs["deterministic"]
            return [1.0, 2.0], [10, 10]

        with patch("hand_imitation.utils.fpo_eval.evaluate_policy", fake_evaluate_policy):
            metrics = callback(model=object(), eval_env=object(), num_timesteps=0)

        self.assertFalse(seen["deterministic"])
        self.assertEqual(metrics["eval/mean_reward"], 1.5)

    def test_rfpo_phase_schedule_selects_latest_started_phase(self):
        from algos.rl.fpo_trainer import _normalise_phase_schedule, _phase_values_for_timestep

        schedule = _normalise_phase_schedule(
            [
                {"name": "late", "start_step": 30, "values": {"learning_rate": 3e-7}},
                {"name": "early", "start_step": 0, "values": {"learning_rate": 8e-6}},
                {"name": "mid", "start_step": 10, "values": {"learning_rate": 1e-6}},
            ]
        )

        self.assertEqual([phase["name"] for phase in schedule], ["early", "mid", "late"])
        self.assertEqual(_phase_values_for_timestep(schedule, 0)["learning_rate"], 8e-6)
        self.assertEqual(_phase_values_for_timestep(schedule, 20)["learning_rate"], 1e-6)
        self.assertEqual(_phase_values_for_timestep(schedule, 40)["learning_rate"], 3e-7)

    def test_rfpo_phase_schedule_rejects_warm_start_fields(self):
        from algos.rl.fpo_trainer import _normalise_phase_schedule

        with self.assertRaisesRegex(ValueError, "ppo_base_checkpoint"):
            _normalise_phase_schedule(
                [
                    {
                        "name": "bad",
                        "start_step": 0,
                        "values": {"ppo_base_checkpoint": "/tmp/ppo.zip"},
                    }
                ]
            )

    def test_rfpo_phase_schedule_updates_trainer_policy_and_optimizer(self):
        from algos.rl.fpo_trainer import FPOStateTrainer

        trainer = object.__new__(FPOStateTrainer)
        trainer.policy = self.make_policy(residual_action_scale=0.1)
        trainer.learning_rate = 1e-5
        trainer.min_learning_rate = 1e-7
        trainer.max_learning_rate = 1e-5
        trainer.num_epochs = 5
        trainer.gaussian_action_std = 0.2
        trainer.clip_range = 0.2
        trainer.actor_objective = "gaussian_ppo"
        trainer.trust_region_mode = "ppo"
        trainer.gaussian_log_std = None
        trainer.flow_actor_lr_scale = 1.0
        trainer.residual_head_lr_scale = 1.0
        trainer.direct_head_lr_scale = 1.0
        trainer.gaussian_log_std_lr_scale = 1.0
        trainer.ppo_base_actor_lr_scale = 0.25
        trainer.ppo_base_critic_lr_scale = 0.0
        trainer.trainable_ppo_base_critic = False
        trainer.optimizer = FPOStateTrainer._make_optimizer(trainer)
        trainer._record_optimizer_lr_scales()
        first_optimizer = trainer.optimizer
        trainer.num_timesteps = 0
        trainer.phase_schedule = [
            {
                "name": "early",
                "start_step": 0,
                "reset_optimizer": False,
                "values": {"learning_rate": 8e-6, "num_epochs": 6},
            },
            {
                "name": "precision",
                "start_step": 100,
                "reset_optimizer": True,
                "values": {
                    "learning_rate": 3.5e-7,
                    "min_learning_rate": 3.5e-7,
                    "max_learning_rate": 3.5e-7,
                    "clip_range": 0.018,
                    "gaussian_action_std": 0.009,
                    "residual_action_scale": 0.006,
                },
            },
        ]
        trainer._active_phase_index = None
        trainer._active_phase_name = ""
        trainer._phase_changed_this_iter = False

        trainer._apply_phase_schedule()
        self.assertIs(trainer.optimizer, first_optimizer)
        self.assertEqual(trainer.num_epochs, 6)
        self.assertAlmostEqual(trainer.learning_rate, 8e-6)

        trainer.num_timesteps = 100
        trainer._apply_phase_schedule()

        self.assertIsNot(trainer.optimizer, first_optimizer)
        self.assertAlmostEqual(trainer.learning_rate, 3.5e-7)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 3.5e-7)
        self.assertAlmostEqual(trainer.clip_range, 0.018)
        self.assertAlmostEqual(trainer.gaussian_action_std, 0.009)
        self.assertAlmostEqual(trainer.policy.residual_action_scale, 0.006)
        self.assertAlmostEqual(trainer.policy.cfg.residual_action_scale, 0.006)

    def test_rfpo_oneline_config_is_from_scratch_without_ppo_or_bc_warmstart(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "algos"
            / "rl"
            / "config"
            / "agent"
            / "rfpo_scratch_oneline.yaml"
        )
        text = config_path.read_text()

        self.assertIn("bc_checkpoint: null", text)
        self.assertIn("bc_anchor_dataset: null", text)
        self.assertIn("ppo_base_checkpoint: null", text)
        self.assertIn("ppo_teacher_checkpoint: null", text)
        self.assertIn("phase_schedule:", text)
        self.assertNotIn("restore_checkpoint.zip", text)
        self.assertNotIn("rl_models_", text)


if __name__ == "__main__":
    unittest.main()
