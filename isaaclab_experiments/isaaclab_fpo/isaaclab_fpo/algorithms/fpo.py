# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import TYPE_CHECKING

from isaaclab_fpo.modules import ActorCritic
from isaaclab_fpo.modules.ema import ExponentialMovingAverage
from isaaclab_fpo.modules.normalizer import EmpiricalNormalization
from isaaclab_fpo.modules.reflow_extensions import map_gap_to_reflow_lambda
from isaaclab_fpo.storage import RolloutStorage

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoAlgorithmCfg


def clamp_ste(x, min=None, max=None):
    clamped = x.clamp(min=min, max=max)
    # forward uses clamped; backward uses identity grad wrt x
    return x + (clamped - x).detach()


class FPO:
    """FPO++ implementation."""

    def __init__(
        self,
        policy: ActorCritic,
        cfg: FpoRslRlPpoAlgorithmCfg,
        device="cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # FPO components
        self.policy: ActorCritic = policy
        self.policy.to(self.device)
        # Create optimizer (AdamW if weight_decay > 0, else Adam)
        if cfg.weight_decay > 0:
            self.optimizer = optim.AdamW(
                self.policy.parameters(),
                lr=cfg.learning_rate,
                betas=cfg.adam_betas,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                self.policy.parameters(),
                lr=cfg.learning_rate,
                betas=cfg.adam_betas,
            )

        # EMA components
        self.ema_decay = cfg.ema_decay
        self.ema_warmup_steps = cfg.ema_warmup_steps
        self.tot_timesteps = 0
        if cfg.ema_decay > 0.0:
            self.ema = ExponentialMovingAverage(
                self.policy.actor, decay=cfg.ema_decay, device=self.device
            )
            print(
                f"EMA enabled with decay={cfg.ema_decay}, warmup_steps={cfg.ema_warmup_steps}"
            )
        else:
            self.ema = None

        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # FPO parameters
        self.clip_param = cfg.clip_param
        self.num_learning_epochs = cfg.num_learning_epochs
        self.num_mini_batches = cfg.num_mini_batches
        self.value_loss_coef = cfg.value_loss_coef
        self.knn_entropy_coef = cfg.knn_entropy_coef
        self.knn_entropy_k = cfg.knn_entropy_k
        self.gamma = cfg.gamma
        self.lam = cfg.lam
        self.max_grad_norm = cfg.max_grad_norm
        self.use_clipped_value_loss = cfg.use_clipped_value_loss
        self.desired_kl = cfg.desired_kl
        self.schedule = cfg.schedule
        self.learning_rate = cfg.learning_rate
        self.normalize_advantage_per_mini_batch = cfg.normalize_advantage_per_mini_batch
        self.normalize_advantage = cfg.normalize_advantage
        self.n_samples_per_action = cfg.n_samples_per_action
        self.cfm_loss_clamp = cfg.cfm_loss_clamp
        self.cfm_loss_clamp_negative_advantages = cfg.cfm_loss_clamp_negative_advantages
        self.cfm_loss_clamp_negative_advantages_max = (
            cfg.cfm_loss_clamp_negative_advantages_max
        )
        self.cfm_diff_clamp_max = cfg.cfm_diff_clamp_max
        self.advantage_clamp = cfg.advantage_clamp
        self.storage_action_noise_std = cfg.storage_action_noise_std
        self.trust_region_mode = cfg.trust_region_mode
        self.reflow_enabled = cfg.reflow_enabled
        self.reflow_loss_coef = cfg.reflow_loss_coef
        self.reflow_n_samples_per_obs = cfg.reflow_n_samples_per_obs
        self.reflow_mode = cfg.reflow_mode
        self.reflow_advantage_threshold = cfg.reflow_advantage_threshold
        self.reflow_use_ema_endpoint = cfg.reflow_use_ema_endpoint
        self.reflow_adaptive_lambda_enabled = cfg.reflow_adaptive_lambda_enabled
        self.reflow_lambda_min = cfg.reflow_lambda_min
        self.reflow_lambda_max = cfg.reflow_lambda_max
        self.reflow_lambda_gap_low = cfg.reflow_lambda_gap_low
        self.reflow_lambda_gap_high = cfg.reflow_lambda_gap_high
        self.reflow_lambda_ema = cfg.reflow_lambda_ema
        self.reflow_lambda_warmup_iters = int(cfg.reflow_lambda_warmup_iters)
        self._adaptive_reflow_coef = float(cfg.reflow_lambda_max)
        self._last_adaptive_reflow_gap = 0.0
        self.theory_metrics_enabled = cfg.theory_metrics_enabled
        self.adaptive_compute_enabled = cfg.adaptive_compute_enabled
        self.adaptive_compute_loss_coef = cfg.adaptive_compute_loss_coef
        self.adaptive_latency_penalty_coef = cfg.adaptive_latency_penalty_coef
        self.random_x0_consistency_enabled = cfg.random_x0_consistency_enabled
        self.random_x0_consistency_coef = cfg.random_x0_consistency_coef
        self.random_x0_consistency_steps = list(cfg.random_x0_consistency_steps)
        self.reflow_teacher_checkpoint = str(cfg.reflow_teacher_checkpoint or "")
        self.teacher_kd_enabled = bool(cfg.teacher_kd_enabled)
        self.teacher_kd_coef = float(cfg.teacher_kd_coef)
        self.teacher_kd_steps = list(cfg.teacher_kd_steps)
        self.teacher_aux_zero_x0_prob = float(cfg.teacher_aux_zero_x0_prob)
        self._teacher_actor: nn.Module | None = None
        self._teacher_obs_normalizer: EmpiricalNormalization | None = None
        self._last_teacher_kd_metrics: dict[str, float] = {}
        if self.reflow_teacher_checkpoint:
            self._load_frozen_teacher(self.reflow_teacher_checkpoint)
        self.update_counter = 0
        self._last_theory_metrics: dict[str, float] = {}
        self._last_adaptive_metrics: dict[str, float] = {}
        self._last_random_x0_metrics: dict[str, float] = {}
        if self.reflow_enabled:
            print(
                f"Rectified Flow reflow enabled: mode={self.reflow_mode}, "
                f"coef={self.reflow_loss_coef}, "
                f"n_samples_per_obs={self.reflow_n_samples_per_obs}, "
                f"ema_endpoint={self.reflow_use_ema_endpoint}, "
                f"adaptive_lambda={self.reflow_adaptive_lambda_enabled}, "
                f"lambda_warmup_iters={self.reflow_lambda_warmup_iters}, "
                f"theory_metrics={self.theory_metrics_enabled}, "
                f"adaptive_compute={self.adaptive_compute_enabled}, "
                f"random_x0_consistency={self.random_x0_consistency_enabled}, "
                f"teacher_kd={self.teacher_kd_enabled}"
            )
        if self._teacher_actor is not None:
            print(
                f"Frozen reflow teacher loaded from {self.reflow_teacher_checkpoint} "
                f"(kd={self.teacher_kd_enabled}, kd_steps={self.teacher_kd_steps}, "
                f"aux_zero_x0_prob={self.teacher_aux_zero_x0_prob})"
            )
        if self.random_x0_consistency_enabled:
            print(
                f"Random-x0 consistency loss enabled: "
                f"coef={self.random_x0_consistency_coef}, "
                f"steps={self.random_x0_consistency_steps}, "
                f"train_flow_x0_mode={getattr(self.policy, 'train_flow_x0_mode', 'random')}"
            )

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            self.device,
            self.n_samples_per_action,
        )

    def act(self, obs, critic_obs):
        # Shape assertions
        assert len(obs.shape) == 2, (
            f"Expected obs shape [num_envs, obs_dim], got {obs.shape}"
        )
        assert len(critic_obs.shape) == 2, (
            f"Expected critic_obs shape [num_envs, critic_obs_dim], got {critic_obs.shape}"
        )
        assert obs.shape[0] == self.storage.num_envs, (
            f"Expected {self.storage.num_envs} envs, got {obs.shape[0]}"
        )

        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()

        # Add noise to stored actions for entropy-like regularization
        if self.storage_action_noise_std > 0:
            noise = self.storage_action_noise_std * torch.randn_like(
                self.transition.actions
            )
            self.transition.actions = self.transition.actions + noise

        # Shape assertions for outputs
        assert self.transition.actions.shape == (
            self.storage.num_envs,
            self.policy.num_actions,
        ), (
            f"Expected actions shape [{self.storage.num_envs}, {self.policy.num_actions}], got {self.transition.actions.shape}"
        )
        assert self.transition.values.shape == (self.storage.num_envs, 1), (
            f"Expected values shape [{self.storage.num_envs}, 1], got {self.transition.values.shape}"
        )

        # FPO stuff
        cfm_loss_eps = torch.randn(
            (self.storage.num_envs, self.n_samples_per_action, self.policy.num_actions),
            device=self.device,
        )

        # Sample uniform timesteps
        uniform_t = torch.rand(
            (self.storage.num_envs, self.n_samples_per_action, 1), device=self.device
        )

        # Apply inverse CDF transform using beta parameter
        # For Beta(1, beta) distribution: F^{-1}(u) = 1 - (1-u)^(1/beta)
        # Scale to [0.005, 0.995] to avoid boundary instabilities at t=0 and t=1
        beta = self.policy.cfm_loss_t_inverse_cdf_beta
        cfm_loss_t = 0.005 + 0.99 * (1.0 - (1.0 - uniform_t) ** (1.0 / beta))

        # Shape assertions for CFM inputs
        assert cfm_loss_eps.shape == (
            self.storage.num_envs,
            self.n_samples_per_action,
            self.policy.num_actions,
        )
        assert cfm_loss_t.shape == (
            self.storage.num_envs,
            self.n_samples_per_action,
            1,
        )

        (
            self.transition.initial_cfm_loss,
            self.transition.x1_pred,
            _x0_pred,
        ) = self.policy.get_cfm_loss(
            obs, self.transition.actions, cfm_loss_eps, cfm_loss_t
        )

        self.transition.initial_cfm_loss = self.transition.initial_cfm_loss.detach()
        self.transition.x1_pred = self.transition.x1_pred.detach()

        # Shape assertions
        assert self.transition.initial_cfm_loss.shape == (
            self.storage.num_envs,
            self.n_samples_per_action,
        )
        assert self.transition.x1_pred.shape == (
            self.storage.num_envs,
            self.n_samples_per_action,
            self.policy.num_actions,
        )

        self.transition.cfm_loss_eps = cfm_loss_eps
        self.transition.cfm_loss_t = cfm_loss_t

        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Shape assertions
        assert rewards.shape == (self.storage.num_envs,), (
            f"Expected rewards shape [{self.storage.num_envs}], got {rewards.shape}"
        )
        assert dones.shape == (self.storage.num_envs,), (
            f"Expected dones shape [{self.storage.num_envs}], got {dones.shape}"
        )

        # Record the rewards and dones
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            assert infos["time_outs"].shape == (self.storage.num_envs,)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _load_frozen_teacher(self, checkpoint_path: str) -> None:
        """Load a frozen baseline actor + its obs normalizer for reflow endpoints / KD."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"reflow_teacher_checkpoint not found: {checkpoint_path}"
            )
        loaded = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_sd = loaded["model_state_dict"]
        actor_sd = {
            key[len("actor.") :]: value
            for key, value in model_sd.items()
            if key.startswith("actor.")
        }
        teacher_actor = copy.deepcopy(self.policy.actor).to(self.device)
        teacher_actor.load_state_dict(actor_sd)
        teacher_actor.eval()
        for param in teacher_actor.parameters():
            param.requires_grad_(False)
        self._teacher_actor = teacher_actor

        obs_dim = int(self.policy.num_actor_obs)
        teacher_norm = EmpiricalNormalization(shape=(obs_dim,)).to(self.device)
        teacher_norm.load_state_dict(loaded["obs_norm_state_dict"])
        teacher_norm.eval()
        self._teacher_obs_normalizer = teacher_norm

    def _to_teacher_obs(
        self, student_obs: torch.Tensor, student_normalizer: nn.Module | None
    ) -> torch.Tensor:
        """Map student-normalized obs into the frozen teacher's normalization."""
        if self._teacher_obs_normalizer is None:
            return student_obs
        with torch.no_grad():
            if student_normalizer is not None and hasattr(student_normalizer, "inverse"):
                raw = student_normalizer.inverse(student_obs)
            else:
                raw = student_obs
            return self._teacher_obs_normalizer(raw)

    def _reflow_endpoint_actor(self) -> nn.Module | None:
        """Frozen baseline actor, detached EMA actor, or None (online actor)."""
        if self._teacher_actor is not None:
            return self._teacher_actor
        if not self.reflow_use_ema_endpoint or self.ema is None:
            return None
        if self.tot_timesteps <= self.ema_warmup_steps:
            return None

        if not hasattr(self, "_reflow_endpoint_actor_module"):
            endpoint = copy.deepcopy(self.policy.actor).to(self.device)
            endpoint.eval()
            for p in endpoint.parameters():
                p.requires_grad_(False)
            self._reflow_endpoint_actor_module = endpoint

        endpoint = self._reflow_endpoint_actor_module
        for name, param in endpoint.named_parameters():
            if name in self.ema.shadow_params:
                param.data.copy_(self.ema.shadow_params[name].to(param.device))
        return endpoint

    def compute_returns(self, last_critic_obs):
        # Shape assertion
        assert (
            len(last_critic_obs.shape) == 2
            and last_critic_obs.shape[0] == self.storage.num_envs
        )

        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        assert last_values.shape == (self.storage.num_envs, 1)

        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=self.normalize_advantage
            and not self.normalize_advantage_per_mini_batch,
        )

    def update(self, obs_normalizer=None, privileged_obs_normalizer=None):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_reflow_loss = 0
        mean_adaptive_compute_loss = 0
        mean_random_x0_consistency_loss = 0
        mean_teacher_kd_loss = 0
        mean_entropy = 0
        mean_kl = 0

        # Gradient norm tracking (kept for metrics, not histograms)
        all_grad_norms_before = []
        all_grad_norms_after = []

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # iterate over batches
        mini_batch_step = 0
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_x1_pred_batch,
            old_cfm_loss_batch,
            old_cfm_loss_eps_batch,
            old_cfm_loss_t_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # Apply advantage clamping
            with torch.no_grad():
                positive_clamp, negative_clamp = self.advantage_clamp
                advantages_batch = advantages_batch.clamp(
                    -negative_clamp, positive_clamp
                )

            # Shape assertions for mini-batch
            assert obs_batch.shape[0] == batch_size
            assert actions_batch.shape == (batch_size, self.policy.num_actions)
            assert (
                target_values_batch.shape
                == advantages_batch.shape
                == returns_batch.shape
                == (batch_size, 1)
            )
            assert old_cfm_loss_batch.shape == (
                batch_size,
                self.n_samples_per_action,
            )
            assert old_cfm_loss_eps_batch.shape == (
                batch_size,
                self.n_samples_per_action,
                self.policy.num_actions,
            )
            assert old_cfm_loss_t_batch.shape == (
                batch_size,
                self.n_samples_per_action,
                1,
            )

            # Use stored samples
            cfm_loss_batch, x1_pred_batch, x0_pred_batch = self.policy.get_cfm_loss(
                obs_batch, actions_batch, old_cfm_loss_eps_batch, old_cfm_loss_t_batch
            )
            assert x1_pred_batch.shape == (
                batch_size,
                self.n_samples_per_action,
                self.policy.num_actions,
            )
            assert x0_pred_batch.shape == (
                batch_size,
                self.n_samples_per_action,
                self.policy.num_actions,
            )
            assert cfm_loss_batch.shape == (
                batch_size,
                self.n_samples_per_action,
            )

            # -- critic
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            assert value_batch.shape == (batch_size, 1)

            # entropy
            if self.knn_entropy_coef > 0:
                entropy_bonus = self._compute_knn_entropy(
                    x0_pred_batch, k=self.knn_entropy_k
                )
            else:
                entropy_bonus = None

            # KL
            if self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = (x1_pred_batch.detach() - old_x1_pred_batch) ** 2
                    kl_mean = kl_mean.mean()

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(
                            kl_mean, op=torch.distributed.ReduceOp.SUM
                        )
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

                    mean_kl += kl_mean.item()

            # Surrogate loss
            assert (
                old_cfm_loss_batch.shape
                == cfm_loss_batch.shape
                == advantages_batch.shape[:-1] + (self.n_samples_per_action,)
                == (batch_size, self.n_samples_per_action)
            )
            assert (
                advantages_batch.shape
                == returns_batch.shape
                == target_values_batch.shape
                == (batch_size, 1)
            )

            # Clamp CFM losses symmetrically
            if self.cfm_loss_clamp > 0:
                old_cfm_loss_batch = torch.clamp(
                    old_cfm_loss_batch, max=self.cfm_loss_clamp
                )
                cfm_loss_batch = torch.clamp(cfm_loss_batch, max=self.cfm_loss_clamp)

            # Clamp current CFM loss for negative advantages only
            # Prevents extreme ratios when the policy tries to avoid bad actions
            if self.cfm_loss_clamp_negative_advantages:
                cfm_loss_batch = torch.where(
                    advantages_batch < 0,
                    cfm_loss_batch.clamp(
                        max=self.cfm_loss_clamp_negative_advantages_max
                    ),
                    cfm_loss_batch,
                )

            # Per-sample log ratios (no averaging before exp)
            # Each of the n_samples gets its own ratio, providing more gradient diversity
            log_ratio = old_cfm_loss_batch - cfm_loss_batch
            log_ratio = clamp_ste(log_ratio, max=self.cfm_diff_clamp_max)
            ratio = torch.exp(log_ratio)
            assert ratio.shape == (
                batch_size,
                self.n_samples_per_action,
            )

            # Surrogate computation
            if self.trust_region_mode == "ppo":
                surrogate = -advantages_batch * ratio
                surrogate_clipped = -advantages_batch * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                assert surrogate.shape == surrogate_clipped.shape
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            elif self.trust_region_mode == "spo":
                surrogate_loss = -torch.mean(
                    ratio * advantages_batch
                    - torch.abs(advantages_batch)
                    / (2.0 * self.clip_param)
                    * (ratio - 1.0) ** 2
                )
            elif self.trust_region_mode == "aspo":
                surrogate = -advantages_batch * ratio
                surrogate_clipped = -advantages_batch * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                ppo_loss = torch.max(surrogate, surrogate_clipped)

                spo_loss = -(
                    ratio * advantages_batch
                    - torch.abs(advantages_batch)
                    / (2.0 * self.clip_param)
                    * (ratio - 1.0) ** 2
                )

                surrogate_loss = torch.where(
                    advantages_batch > 0, ppo_loss, spo_loss
                ).mean()
            else:
                raise ValueError(f"Unknown trust_region_mode: {self.trust_region_mode}")

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            reflow_loss = torch.tensor(0.0, device=self.device)
            if self.reflow_enabled:
                if self.reflow_adaptive_lambda_enabled and mini_batch_step == 0:
                    gap_obs = obs_batch[: min(128, obs_batch.shape[0])]
                    gap = self.policy.compute_random_x0_step_gap(gap_obs, low_steps=1)
                    self._last_adaptive_reflow_gap = gap
                    if self.update_counter >= self.reflow_lambda_warmup_iters:
                        target = map_gap_to_reflow_lambda(
                            gap,
                            self.reflow_lambda_min,
                            self.reflow_lambda_max,
                            self.reflow_lambda_gap_low,
                            self.reflow_lambda_gap_high,
                        )
                        ema = self.reflow_lambda_ema
                        self._adaptive_reflow_coef = (
                            ema * self._adaptive_reflow_coef + (1.0 - ema) * target
                        )
                # IMPORTANT: do NOT copy EMA weights onto the training actor.
                # Endpoints must come from a detached EMA copy; CFM prediction stays
                # on the online actor. The old store/copy/restore path made both use
                # EMA weights, so reflow grads were computed under EMA and then applied
                # after restore — this destabilized fpo_operator (policy collapse).
                reflow_loss = self.policy.get_reflow_loss(
                    obs_batch,
                    self.reflow_n_samples_per_obs,
                    advantages=advantages_batch
                    if self.reflow_mode in {"reward_aware", "fpo_operator"}
                    else None,
                    reflow_mode=self.reflow_mode,
                    advantage_threshold=self.reflow_advantage_threshold,
                    endpoint_actor=self._reflow_endpoint_actor(),
                    endpoint_obs=self._to_teacher_obs(obs_batch, obs_normalizer)
                    if self._teacher_actor is not None
                    else None,
                    zero_x0_prob=self.teacher_aux_zero_x0_prob
                    if self._teacher_actor is not None
                    else 0.0,
                )

                if self.theory_metrics_enabled and mini_batch_step == 0:
                    self._last_theory_metrics = self.policy.compute_theory_metrics(
                        obs_batch[: min(256, obs_batch.shape[0])]
                    )

            adaptive_loss = torch.tensor(0.0, device=self.device)
            if self.adaptive_compute_enabled:
                adaptive_loss, self._last_adaptive_metrics = (
                    self.policy.get_adaptive_compute_loss(
                        obs_batch,
                        advantages=advantages_batch,
                        latency_penalty_coef=self.adaptive_latency_penalty_coef,
                    )
                )

            random_x0_loss = torch.tensor(0.0, device=self.device)
            if self.random_x0_consistency_enabled:
                random_x0_loss, self._last_random_x0_metrics = (
                    self.policy.get_random_x0_consistency_loss(
                        obs_batch,
                        step_bins=self.random_x0_consistency_steps,
                    )
                )

            teacher_kd_loss = torch.tensor(0.0, device=self.device)
            if self.teacher_kd_enabled:
                if self._teacher_actor is None:
                    raise RuntimeError(
                        "teacher_kd_enabled requires reflow_teacher_checkpoint"
                    )
                teacher_kd_loss, self._last_teacher_kd_metrics = (
                    self.policy.get_teacher_kd_loss(
                        obs_batch,
                        self._to_teacher_obs(obs_batch, obs_normalizer),
                        teacher_actor=self._teacher_actor,
                        step_bins=self.teacher_kd_steps,
                        zero_x0_prob=self.teacher_aux_zero_x0_prob,
                    )
                )

            loss = surrogate_loss + self.value_loss_coef * value_loss
            if self.reflow_enabled:
                reflow_coef = (
                    self._adaptive_reflow_coef
                    if self.reflow_adaptive_lambda_enabled
                    else self.reflow_loss_coef
                )
                loss = loss + reflow_coef * reflow_loss
            if self.adaptive_compute_enabled:
                loss = loss + self.adaptive_compute_loss_coef * adaptive_loss
            if self.random_x0_consistency_enabled:
                loss = loss + self.random_x0_consistency_coef * random_x0_loss
            if self.teacher_kd_enabled:
                loss = loss + self.teacher_kd_coef * teacher_kd_loss
            if entropy_bonus is not None:
                loss -= self.knn_entropy_coef * entropy_bonus

            # Compute the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Track gradient norms before clipping
            total_grad_norm_before = 0.0
            for p in self.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm_before += param_norm.item() ** 2
            total_grad_norm_before = total_grad_norm_before**0.5

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            # Track gradient norms after clipping
            total_grad_norm_after = 0.0
            for p in self.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm_after += param_norm.item() ** 2
            total_grad_norm_after = total_grad_norm_after**0.5

            self.optimizer.step()

            # Store gradient norms for debugging
            all_grad_norms_before.append(total_grad_norm_before)
            all_grad_norms_after.append(total_grad_norm_after)

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            if self.reflow_enabled:
                mean_reflow_loss += reflow_loss.item()
            if self.adaptive_compute_enabled:
                mean_adaptive_compute_loss += adaptive_loss.item()
            if self.random_x0_consistency_enabled:
                mean_random_x0_consistency_loss += random_x0_loss.item()
            if self.teacher_kd_enabled:
                mean_teacher_kd_loss += teacher_kd_loss.item()
            mean_entropy += entropy_bonus.item() if entropy_bonus is not None else 0.0

            mini_batch_step += 1

        # -- Averages
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        if self.reflow_enabled:
            mean_reflow_loss /= num_updates
        if self.adaptive_compute_enabled:
            mean_adaptive_compute_loss /= num_updates
        if self.random_x0_consistency_enabled:
            mean_random_x0_consistency_loss /= num_updates
        if self.teacher_kd_enabled:
            mean_teacher_kd_loss /= num_updates
        mean_entropy /= num_updates
        if self.schedule == "adaptive":
            mean_kl /= num_updates
        self.storage.clear()

        # Increment counters
        self.update_counter += 1
        self.tot_timesteps += 1

        # construct the loss dictionary (actual losses only)
        loss_dict = {
            "surrogate_loss": mean_surrogate_loss,
            "value_loss": mean_value_loss,
        }
        if self.reflow_enabled:
            loss_dict["reflow_loss"] = mean_reflow_loss
            if self.reflow_adaptive_lambda_enabled:
                loss_dict["adaptive_reflow_coef"] = self._adaptive_reflow_coef
                loss_dict["adaptive_reflow_gap"] = self._last_adaptive_reflow_gap
        if self.adaptive_compute_enabled:
            loss_dict["adaptive_compute_loss"] = mean_adaptive_compute_loss
            for key, value in self._last_adaptive_metrics.items():
                loss_dict[key] = value
        if self.random_x0_consistency_enabled:
            loss_dict["random_x0_consistency_loss"] = mean_random_x0_consistency_loss
            for key, value in self._last_random_x0_metrics.items():
                if key != "random_x0_consistency_loss":
                    loss_dict[key] = value
        if self.teacher_kd_enabled:
            loss_dict["teacher_kd_loss"] = mean_teacher_kd_loss
            for key, value in self._last_teacher_kd_metrics.items():
                if key != "teacher_kd_loss":
                    loss_dict[key] = value
        if self.theory_metrics_enabled:
            for key, value in self._last_theory_metrics.items():
                loss_dict[key] = value
        if self.knn_entropy_coef > 0:
            loss_dict["entropy_loss"] = mean_entropy

        # construct the metrics dictionary (non-loss metrics)
        metrics_dict = {
            "clip_param": self.clip_param,
        }
        if self.schedule == "adaptive":
            metrics_dict["kl"] = mean_kl

        # Gradient norm metrics (scalar, not histograms)
        if all_grad_norms_before:
            metrics_dict["mean_grad_norm_before_clip"] = np.mean(all_grad_norms_before)
            metrics_dict["mean_grad_norm_after_clip"] = np.mean(all_grad_norms_after)

        # Observation normalizer scalar statistics
        if obs_normalizer is not None:
            with torch.no_grad():
                obs_std = obs_normalizer.std.cpu()
                metrics_dict["obs_norm_min_std"] = obs_std.min().item()
                metrics_dict["obs_norm_max_std"] = obs_std.max().item()
                metrics_dict["obs_norm_mean_std"] = obs_std.mean().item()

        if privileged_obs_normalizer is not None:
            with torch.no_grad():
                priv_obs_std = privileged_obs_normalizer.std.cpu()
                metrics_dict["privileged_obs_norm_min_std"] = priv_obs_std.min().item()
                metrics_dict["privileged_obs_norm_max_std"] = priv_obs_std.max().item()
                metrics_dict["privileged_obs_norm_mean_std"] = (
                    priv_obs_std.mean().item()
                )

        # Add metrics to loss_dict under "metrics" key
        loss_dict["metrics"] = metrics_dict

        return loss_dict

    """
    Entropy computation methods
    """

    def _compute_knn_entropy(self, x0_pred: torch.Tensor, k: int) -> torch.Tensor:
        """Compute k-NN entropy estimate.

        Non-parametric entropy estimator based on nearest neighbor distances.
        Naturally handles multimodal distributions.

        Args:
            x0_pred: Predicted actions [batch, n_samples, action_dim]
            k: Number of nearest neighbors (typically 1)

        Returns:
            Scalar entropy estimate (in nats)
        """
        batch_size, n_samples, action_dim = x0_pred.shape
        assert k >= 1 and k < n_samples, (
            f"k must be in [1, n_samples), got k={k}, n_samples={n_samples}"
        )

        # Compute pairwise distances: [batch, n_samples, n_samples]
        dists = torch.cdist(x0_pred, x0_pred, p=2)

        # Set diagonal to large value to exclude self-distances
        eye_mask = (
            torch.eye(n_samples, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        dists = dists + eye_mask * 1e10

        # Find k-th nearest neighbor distance
        kth_dists, _ = torch.topk(dists, k=k, dim=2, largest=False, sorted=True)
        rho_k = kth_dists[:, :, -1]

        # Clamp distances for numerical stability
        rho_k = torch.clamp(rho_k, min=1e-6, max=1e9)

        # Kozachenko-Leonenko estimator
        psi_n = torch.digamma(torch.tensor(float(n_samples), device=self.device))
        psi_k = torch.digamma(torch.tensor(float(k), device=self.device))
        log_cd = (action_dim / 2) * np.log(np.pi) - float(
            torch.lgamma(torch.tensor(action_dim / 2 + 1))
        )
        log_rho_mean = torch.log(rho_k).mean(dim=1)
        entropy_per_batch = psi_n - psi_k + log_cd + action_dim * log_rho_mean

        return entropy_per_batch.mean()

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them."""
        grads = [
            param.grad.view(-1)
            for param in self.policy.parameters()
            if param.grad is not None
        ]
        all_grads = torch.cat(grads)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(
                    all_grads[offset : offset + numel].view_as(param.grad.data)
                )
                offset += numel
