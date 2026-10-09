# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

from isaaclab_fpo.utils import resolve_nn_activation
from isaaclab_fpo.modules.reflow_extensions import (
    StepPredictor,
    compute_advantage_reflow_weights,
    compute_discretization_gap,
    compute_path_straightness,
)

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cfg: FpoRslRlPpoActorCriticCfg,
    ):
        super().__init__()
        activation = resolve_nn_activation(cfg.activation)

        # Policy parameters
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.timestep_embed_dim = cfg.timestep_embed_dim
        self.mlp_output_scale = cfg.actor_mlp_output_scale
        self.cfm_loss_t_inverse_cdf_beta = cfg.cfm_loss_t_inverse_cdf_beta
        self.sampling_steps = cfg.sampling_steps
        self.cfm_loss_reduction = cfg.cfm_loss_reduction

        # Inference parameters
        self.actor_scale = cfg.actor_scale

        # Training parameters
        self.action_perturb_std = cfg.action_perturb_std
        self.train_flow_x0_mode = cfg.train_flow_x0_mode
        self.train_flow_x0_random_prob = float(cfg.train_flow_x0_random_prob)
        if cfg.training_sampling_steps is not None:
            self.training_sampling_steps = cfg.training_sampling_steps
        else:
            self.training_sampling_steps = cfg.sampling_steps

        # Policy Network: Actor
        actor_hidden_dims = cfg.actor_hidden_dims
        critic_hidden_dims = cfg.critic_hidden_dims
        mlp_input_dim_a = num_actor_obs + self.timestep_embed_dim + num_actions
        mlp_input_dim_c = num_critic_obs
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[layer_index], num_actions)
                )
            else:
                actor_layers.append(
                    nn.Linear(
                        actor_hidden_dims[layer_index],
                        actor_hidden_dims[layer_index + 1],
                    )
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Apply scaling to actor's final layer weights
        if (
            cfg.actor_final_layer_weight_scale is not None
            and cfg.actor_final_layer_weight_scale != 1.0
        ):
            final_layer = self.actor[-1]
            assert isinstance(final_layer, nn.Linear), (
                "Expected final layer to be Linear"
            )
            with torch.no_grad():
                final_layer.weight.data *= cfg.actor_final_layer_weight_scale
                if final_layer.bias is not None:
                    final_layer.bias.data *= cfg.actor_final_layer_weight_scale
            print(
                f"Applied actor_final_layer_weight_scale={cfg.actor_final_layer_weight_scale} to final layer"
            )

        # Policy Network: Critic
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(
                    nn.Linear(
                        critic_hidden_dims[layer_index],
                        critic_hidden_dims[layer_index + 1],
                    )
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.adaptive_compute_enabled = cfg.adaptive_compute_enabled
        self.adaptive_step_bins = list(cfg.adaptive_step_bins)
        if self.adaptive_compute_enabled:
            self.step_predictor = StepPredictor(
                num_obs=num_actor_obs,
                step_bins=self.adaptive_step_bins,
            )
            print(
                f"StepPredictor enabled with bins={self.adaptive_step_bins}"
            )
        else:
            self.step_predictor = None

        # Compile the inner flow integration loop for CUDA graph replay.
        # Cached CUDA graph can lead to a 3~9x speedup.
        self._compiled_integrate_flow = torch.compile(
            self._integrate_flow, mode="reduce-overhead"
        )

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def act(self, observations: torch.Tensor, **kwargs):
        device = observations.device

        assert len(observations.shape) == 2, (
            "observations should be of shape (batch_size, obs_dim)"
        )
        batch_size = observations.shape[0]

        if not self.training:
            x_t = torch.zeros(size=(batch_size, self.num_actions), device=device)
        else:
            x_t = self._sample_train_flow_x0(batch_size, device)

        flow_steps = self.sampling_steps
        full_t_path = torch.linspace(1.0, 0.0, flow_steps + 1, device=device)
        t_current = full_t_path[:-1]
        t_next = full_t_path[1:]
        dt = t_next - t_current

        # Use compiled integration loop for CUDA graph replay speedup
        x_t = self._compiled_integrate_flow(
            observations, x_t, t_current, dt, flow_steps
        )
        actions = self.actor_scale * x_t

        # Perturb action with random noise, this can be interpreted as an entropy regularizer
        if self.training and self.action_perturb_std > 0:
            noise = self.action_perturb_std * torch.randn_like(actions)
            actions = actions + noise

        return actions

    def get_cfm_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        actor: torch.nn.Module | None = None,
    ):
        """Compute CFM loss for training.

        Returns:
            Tuple of (loss, x1_pred, x0_pred)
        """
        # Use provided actor or default to self.actor
        if actor is None:
            actor = self.actor

        (batch_dims, action_dim) = actions.shape
        assert len(observations.shape) == 2, (
            "observations should be of shape (batch_size, obs_dim)"
        )
        assert observations.shape[0] == batch_dims, (
            "actor_obs and actions should have the same batch size"
        )

        # Scale actions to match the scaled action space used during inference
        # During inference, we output self.actor_scale * x_t, so during training
        # we need to learn flow in the same scaled space
        scaled_actions = actions / self.actor_scale

        # Naive velocity MSE loss (hardcoded "u" mode)
        n_samples_per_action = eps.shape[1]
        assert eps.shape == (batch_dims, n_samples_per_action, action_dim)
        assert t.shape == (batch_dims, n_samples_per_action, 1)

        # Compute the embedded timestep
        embedded_t = self._embed_timestep(t)
        x_t = t * eps + (1.0 - t) * scaled_actions[:, None, :]
        # Broadcast actor_obs to match the batch shape
        actor_obs_expanded = observations[:, None, :].expand(
            batch_dims, n_samples_per_action, -1
        )
        # Handle flow network output parameterization (hardcoded to "u" mode)
        mlp_output = actor(torch.cat([actor_obs_expanded, embedded_t, x_t], dim=-1))
        mlp_output = self.mlp_output_scale * mlp_output  # Scale MLP output

        # Direct velocity prediction (u mode)
        velocity_pred = mlp_output
        x0_pred = x_t - t * velocity_pred
        x1_pred = x0_pred + velocity_pred

        # Target velocity is eps - scaled_actions (true flow velocity in scaled space)
        target_velocity = eps - scaled_actions[:, None, :]
        loss = self._compute_squared_error(velocity_pred, target_velocity)
        assert loss.shape == (batch_dims, n_samples_per_action)

        return loss, x1_pred, x0_pred

    def get_reflow_loss(
        self,
        observations: torch.Tensor,
        n_samples_per_obs: int,
        advantages: torch.Tensor | None = None,
        reflow_mode: str = "uniform",
        advantage_threshold: float = 0.0,
        endpoint_actor: torch.nn.Module | None = None,
        endpoint_obs: torch.Tensor | None = None,
        zero_x0_prob: float = 0.0,
    ) -> torch.Tensor:
        """Rectified Flow reflow loss on straight paths between noise and flow endpoints."""
        assert len(observations.shape) == 2
        batch_size, action_dim = observations.shape[0], self.num_actions
        device = observations.device
        endpoint_obs = observations if endpoint_obs is None else endpoint_obs
        assert endpoint_obs.shape == observations.shape

        x0 = torch.randn(
            batch_size, n_samples_per_obs, action_dim, device=device
        )
        if zero_x0_prob > 0.0:
            zero_mask = (
                torch.rand(batch_size, n_samples_per_obs, 1, device=device)
                < zero_x0_prob
            )
            x0 = torch.where(zero_mask, torch.zeros_like(x0), x0)
        flat_endpoint_obs = endpoint_obs[:, None, :].expand(
            batch_size, n_samples_per_obs, -1
        ).reshape(batch_size * n_samples_per_obs, -1)
        flat_x0 = x0.reshape(batch_size * n_samples_per_obs, action_dim)

        integration_actor = endpoint_actor if endpoint_actor is not None else self.actor
        with torch.no_grad():
            t_current, dt, flow_steps = self._flow_integration_grid(
                device, self.sampling_steps
            )
            x1_prime = self._integrate_flow_with_actor(
                integration_actor,
                flat_endpoint_obs,
                flat_x0,
                t_current,
                dt,
                flow_steps,
            )
        x1_prime = x1_prime.reshape(batch_size, n_samples_per_obs, action_dim)

        uniform_t = torch.rand(
            batch_size, n_samples_per_obs, 1, device=device
        )
        beta = self.cfm_loss_t_inverse_cdf_beta
        t = 0.005 + 0.99 * (1.0 - (1.0 - uniform_t) ** (1.0 / beta))

        x_t = t * x0 + (1.0 - t) * x1_prime
        target_velocity = x0 - x1_prime

        embedded_t = self._embed_timestep(t)
        obs_expanded = observations[:, None, :].expand(
            batch_size, n_samples_per_obs, -1
        )
        mlp_output = self.actor(
            torch.cat([obs_expanded, embedded_t, x_t], dim=-1)
        )
        velocity_pred = self.mlp_output_scale * mlp_output

        per_sample_loss = self._compute_squared_error(
            velocity_pred, target_velocity
        )

        if reflow_mode in {"reward_aware", "fpo_operator"} and advantages is not None:
            sample_weights = compute_advantage_reflow_weights(
                advantages, reflow_mode, advantage_threshold
            )
            obs_weights = sample_weights[:, None].expand_as(per_sample_loss)
            return (obs_weights * per_sample_loss).sum() / (
                obs_weights.sum() + 1e-8
            )
        return per_sample_loss.mean()

    def _sample_train_flow_x0(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Sample flow noise for on-policy rollouts according to ``train_flow_x0_mode``."""
        shape = (batch_size, self.num_actions)
        if self.train_flow_x0_mode == "zero":
            return torch.zeros(size=shape, device=device)
        if self.train_flow_x0_mode == "random":
            return torch.randn(size=shape, device=device)
        if self.train_flow_x0_mode == "mix":
            random_x0 = torch.randn(size=shape, device=device)
            zero_x0 = torch.zeros(size=shape, device=device)
            mask = (
                torch.rand(batch_size, 1, device=device)
                < self.train_flow_x0_random_prob
            )
            return torch.where(mask, random_x0, zero_x0)
        raise ValueError(f"Unknown train_flow_x0_mode: {self.train_flow_x0_mode}")

    def get_random_x0_consistency_loss(
        self,
        observations: torch.Tensor,
        step_bins: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Match few-step actions to full-step actions under a shared random x0.

        Unlike adaptive_compute (which uses zero init + a step predictor), this loss
        always starts from x0 ~ N(0, I) — the RF hypothesis / PostEval ``random`` mode.
        """
        batch_size = observations.shape[0]
        device = observations.device
        bins = list(step_bins) if step_bins is not None else [1, 4, 8]
        bins = [int(s) for s in bins if 0 < int(s) < self.sampling_steps]
        if not bins:
            zero = torch.tensor(0.0, device=device)
            return zero, {}

        x0 = torch.randn(batch_size, self.num_actions, device=device)
        with torch.no_grad():
            t_full, dt_full, steps_full = self._flow_integration_grid(
                device, self.sampling_steps
            )
            action_full = self._integrate_flow(
                observations, x0, t_full, dt_full, steps_full
            )

        total_loss = torch.tensor(0.0, device=device)
        metrics: dict[str, float] = {}
        for steps in bins:
            t_k, dt_k, flow_steps = self._flow_integration_grid(device, steps)
            action_low = self._integrate_flow(
                observations, x0, t_k, dt_k, flow_steps
            )
            step_loss = (action_low - action_full).pow(2).mean()
            total_loss = total_loss + step_loss
            metrics[f"random_x0_consistency_{steps}"] = step_loss.item()

        loss = total_loss / float(len(bins))
        metrics["random_x0_consistency_loss"] = loss.item()
        return loss, metrics

    def get_teacher_kd_loss(
        self,
        student_obs: torch.Tensor,
        teacher_obs: torch.Tensor,
        teacher_actor: torch.nn.Module,
        step_bins: list[int] | None = None,
        zero_x0_prob: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Match student few-step actions to a frozen teacher's 64-step action.

        Same x0 for teacher and student. Teacher endpoints are stopgrad.
        """
        assert student_obs.shape == teacher_obs.shape
        batch_size = student_obs.shape[0]
        device = student_obs.device
        bins = list(step_bins) if step_bins is not None else [1, 4, 8]
        bins = [int(s) for s in bins if 0 < int(s) < self.sampling_steps]
        if not bins:
            zero = torch.tensor(0.0, device=device)
            return zero, {}

        x0 = torch.randn(batch_size, self.num_actions, device=device)
        if zero_x0_prob > 0.0:
            zero_mask = torch.rand(batch_size, 1, device=device) < zero_x0_prob
            x0 = torch.where(zero_mask, torch.zeros_like(x0), x0)

        with torch.no_grad():
            t_full, dt_full, steps_full = self._flow_integration_grid(
                device, self.sampling_steps
            )
            action_teacher = self._integrate_flow_with_actor(
                teacher_actor, teacher_obs, x0, t_full, dt_full, steps_full
            )

        total_loss = torch.tensor(0.0, device=device)
        metrics: dict[str, float] = {}
        for steps in bins:
            t_k, dt_k, flow_steps = self._flow_integration_grid(device, steps)
            action_low = self._integrate_flow(
                student_obs, x0, t_k, dt_k, flow_steps
            )
            step_loss = (action_low - action_teacher).pow(2).mean()
            total_loss = total_loss + step_loss
            metrics[f"teacher_kd_{steps}"] = step_loss.item()

        loss = total_loss / float(len(bins))
        metrics["teacher_kd_loss"] = loss.item()
        return loss, metrics

    def get_adaptive_compute_loss(
        self,
        observations: torch.Tensor,
        advantages: torch.Tensor | None = None,
        latency_penalty_coef: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Train step predictor: match low-step actions to full-step actions with latency penalty."""
        if self.step_predictor is None:
            zero = torch.tensor(0.0, device=observations.device)
            return zero, {}

        batch_size = observations.shape[0]
        device = observations.device
        x0 = torch.zeros(batch_size, self.num_actions, device=device)

        with torch.no_grad():
            t_full, dt_full, steps_full = self._flow_integration_grid(
                device, self.sampling_steps
            )
            action_full = self._integrate_flow(
                observations, x0, t_full, dt_full, steps_full
            )

        pred_steps = self.step_predictor.predict_steps(observations)
        action_low = torch.zeros_like(action_full)
        for steps in self.adaptive_step_bins:
            mask = pred_steps == steps
            if not mask.any():
                continue
            t_k, dt_k, flow_steps = self._flow_integration_grid(device, int(steps))
            action_low[mask] = self._integrate_flow(
                observations[mask], x0[mask], t_k, dt_k, flow_steps
            )

        fidelity_loss = (action_low - action_full).pow(2).mean()
        latency_penalty = self.step_predictor.expected_step_fraction(observations).mean()
        adv_bonus = torch.tensor(0.0, device=device)
        if advantages is not None:
            adv_bonus = torch.clamp(advantages.reshape(-1), min=0.0).mean()

        loss = fidelity_loss + latency_penalty_coef * latency_penalty - 0.1 * adv_bonus
        metrics = {
            "adaptive_fidelity_loss": fidelity_loss.item(),
            "adaptive_latency_penalty": latency_penalty.item(),
            "adaptive_mean_steps": pred_steps.float().mean().item(),
        }
        return loss, metrics

    @torch.no_grad()
    def compute_theory_metrics(
        self,
        observations: torch.Tensor,
        n_samples_per_obs: int = 1,
    ) -> dict[str, float]:
        """Theory hooks: straightness and discretization error proxies."""
        batch_size, action_dim = observations.shape[0], self.num_actions
        device = observations.device
        x0 = torch.zeros(batch_size, n_samples_per_obs, action_dim, device=device)
        flat_obs = observations[:, None, :].expand(
            batch_size, n_samples_per_obs, -1
        ).reshape(batch_size * n_samples_per_obs, -1)
        flat_x0 = x0.reshape(batch_size * n_samples_per_obs, action_dim)

        t_full, dt_full, steps_full = self._flow_integration_grid(
            device, self.sampling_steps
        )
        x1 = self._integrate_flow(flat_obs, flat_x0, t_full, dt_full, steps_full)
        x1 = x1.reshape(batch_size, n_samples_per_obs, action_dim)

        t_mid = torch.full((batch_size, n_samples_per_obs, 1), 0.5, device=device)
        x_t = t_mid * x0 + (1.0 - t_mid) * x1
        target_velocity = x0 - x1
        embedded_t = self._embed_timestep(t_mid)
        obs_expanded = observations[:, None, :].expand(
            batch_size, n_samples_per_obs, -1
        )
        velocity_pred = self.mlp_output_scale * self.actor(
            torch.cat([obs_expanded, embedded_t, x_t], dim=-1)
        )

        straightness = compute_path_straightness(
            x0, x1, velocity_pred
        ).mean().item()

        metrics = {"path_straightness": straightness}
        for low_steps in (1, 4, 8, 16):
            if low_steps >= self.sampling_steps:
                continue
            t_k, dt_k, flow_steps = self._flow_integration_grid(device, low_steps)
            x1_low = self._integrate_flow(
                flat_obs, flat_x0, t_k, dt_k, flow_steps
            ).reshape(batch_size, n_samples_per_obs, action_dim)
            gap = compute_discretization_gap(x1, x1_low).mean().item()
            metrics[f"discretization_gap_{low_steps}"] = gap
        return metrics

    @torch.no_grad()
    def compute_random_x0_step_gap(
        self,
        observations: torch.Tensor,
        low_steps: int = 1,
    ) -> float:
        """RMS ||a_N - a_k|| under x0 ~ N(0,I), matching baseline collection/eval-random."""
        batch_size = observations.shape[0]
        device = observations.device
        x0 = torch.randn(batch_size, self.num_actions, device=device)
        t_full, dt_full, steps_full = self._flow_integration_grid(
            device, self.sampling_steps
        )
        action_full = self._integrate_flow(
            observations, x0, t_full, dt_full, steps_full
        )
        t_k, dt_k, flow_steps = self._flow_integration_grid(device, low_steps)
        action_low = self._integrate_flow(
            observations, x0, t_k, dt_k, flow_steps
        )
        return compute_discretization_gap(action_full, action_low).mean().item()

    def _integrate_flow_with_actor(
        self,
        actor: torch.nn.Module,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        dt: torch.Tensor,
        flow_steps: int,
    ) -> torch.Tensor:
        """Flow integration using an alternate actor module (e.g., EMA snapshot)."""
        batch_size = observations.shape[0]
        half_dim = self.timestep_embed_dim // 2
        freqs = 2 ** torch.arange(
            half_dim, device=observations.device, dtype=observations.dtype
        )
        for i in range(flow_steps):
            t_val = t_current[i].reshape(1, 1)
            scaled_t = t_val * freqs
            embedded_t = torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)
            embedded_t = embedded_t.expand(batch_size, -1)
            mlp_output = actor(torch.cat([observations, embedded_t, x_t], dim=-1))
            mlp_output = self.mlp_output_scale * mlp_output
            x_t = x_t + mlp_output * dt[i]
        return x_t

    def get_reflow_loss_legacy(
        self,
        observations: torch.Tensor,
        n_samples_per_obs: int,
    ) -> torch.Tensor:
        """Legacy uniform reflow loss (wrapper)."""
        return self.get_reflow_loss(
            observations, n_samples_per_obs, reflow_mode="uniform"
        )

    def _flow_integration_grid(
        self, device: torch.device, flow_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (t_current, dt, flow_steps) for Euler integration t: 1 -> 0."""
        full_t_path = torch.linspace(1.0, 0.0, flow_steps + 1, device=device)
        t_current = full_t_path[:-1]
        dt = full_t_path[1:] - full_t_path[:-1]
        return t_current, dt, flow_steps

    def _embed_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Embed (*, 1) timestep into (*, timestep_embed_dim)."""
        assert t.shape[-1] == 1
        freqs = 2 ** torch.arange(self.timestep_embed_dim // 2, device=t.device)
        scaled_t = t * freqs
        out = torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)
        assert out.shape == (*t.shape[:-1], self.timestep_embed_dim)
        return out

    def _integrate_flow(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        dt: torch.Tensor,
        flow_steps: int,
    ) -> torch.Tensor:
        """Inner flow integration loop extracted for torch.compile.

        This method contains only static-shape tensor operations and constant
        control flow (hardcoded to "u" mode velocity prediction),
        making it safe for CUDA graph capture via torch.compile(mode="reduce-overhead").

        Args:
            observations: (batch_size, obs_dim) observation tensor.
            x_t: (batch_size, num_actions) initial noise / sample.
            t_current: (flow_steps,) current timestep values.
            dt: (flow_steps,) timestep deltas.
            flow_steps: Number of integration steps (must be constant across calls).

        Returns:
            x_t: (batch_size, num_actions) integrated sample (denoised actions).
        """
        batch_size = observations.shape[0]
        half_dim = self.timestep_embed_dim // 2
        freqs = 2 ** torch.arange(
            half_dim, device=observations.device, dtype=observations.dtype
        )

        for i in range(flow_steps):
            # Inline timestep embedding (avoids assert overhead in compiled path)
            t_val = t_current[i].reshape(1, 1)
            scaled_t = t_val * freqs  # (1, half_dim)
            embedded_t = torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)
            embedded_t = embedded_t.expand(batch_size, -1)

            # Forward through actor network
            mlp_output = self.actor(torch.cat([observations, embedded_t, x_t], dim=-1))
            mlp_output = self.mlp_output_scale * mlp_output

            # Compute velocity from network output (hardcoded to "u" mode)
            u = mlp_output
            x_t = x_t + u * dt[i]

        return x_t

    def _compute_squared_error(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute squared error with configurable reduction."""
        if self.cfm_loss_reduction == "mean":
            return torch.mean((predictions - targets) ** 2, dim=-1)
        elif self.cfm_loss_reduction == "sum":
            return torch.sum((predictions - targets) ** 2, dim=-1)
        else:  # "sqrt"
            squared_errors = (predictions - targets) ** 2
            return torch.sum(squared_errors, dim=-1) / (
                squared_errors.shape[-1] ** 0.5
            )

    def act_inference(self, observations, eval_mode="zero", eval_fixed_seed=12345):
        """Inference with configurable deterministic sampling for flow matching.

        Args:
            observations: Input observations
            eval_mode: Sampling strategy for initial noise
                - "zero": Use zeros for initial noise
                - "fixed_seed": Use fixed seed for reproducible noise
                - "random": Use random noise (different each time)
            eval_fixed_seed: Random seed for fixed_seed mode

        Returns:
            Actions tensor
        """
        device = observations.device
        assert len(observations.shape) == 2, (
            "observations should be of shape (batch_size, obs_dim)"
        )
        batch_size = observations.shape[0]

        # Initialize x_t based on eval_mode
        if eval_mode == "zero":
            x_t = torch.zeros(size=(batch_size, self.num_actions), device=device)
        elif eval_mode == "fixed_seed":
            generator = torch.Generator(device=device)
            generator.manual_seed(eval_fixed_seed)
            x_t = torch.randn(
                size=(batch_size, self.num_actions), device=device, generator=generator
            )
        elif eval_mode == "random":
            x_t = torch.randn(size=(batch_size, self.num_actions), device=device)
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

        flow_steps = self.sampling_steps
        full_t_path = torch.linspace(1.0, 0.0, flow_steps + 1, device=device)
        t_current = full_t_path[:-1]
        t_next = full_t_path[1:]
        dt = t_next - t_current

        # Use compiled integration loop for CUDA graph replay speedup
        x_t = self._compiled_integrate_flow(
            observations, x_t, t_current, dt, flow_steps
        )

        actions = self.actor_scale * x_t
        return actions

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
