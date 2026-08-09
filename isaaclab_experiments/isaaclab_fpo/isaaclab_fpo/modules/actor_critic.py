# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

from isaaclab_fpo.utils import resolve_nn_activation

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
        self.integration_method = cfg.integration_method
        if self.integration_method not in {"euler", "midpoint"}:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
        self.cfm_loss_reduction = cfg.cfm_loss_reduction

        # Inference parameters
        self.actor_scale = cfg.actor_scale

        # Training parameters
        self.action_perturb_std = cfg.action_perturb_std
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

        # Compile the inner flow integration loop for CUDA graph replay.
        # Cached CUDA graph can lead to a 3~9x speedup.
        self._compiled_integrate_flow = torch.compile(
            self._integrate_flow, mode="reduce-overhead"
        )
        self._compiled_integrate_flow_midpoint = torch.compile(
            self._integrate_flow_midpoint, mode="reduce-overhead"
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
            x_t = torch.randn(size=(batch_size, self.num_actions), device=device)

        flow_steps = self.sampling_steps
        full_t_path = torch.linspace(1.0, 0.0, flow_steps + 1, device=device)
        t_current = full_t_path[:-1]
        t_next = full_t_path[1:]
        dt = t_next - t_current

        x_t = self._integrate_selected_flow(
            observations, x_t, t_current, dt, flow_steps, self.integration_method
        )

        # Scale actions
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

    def _integrate_flow_midpoint(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        dt: torch.Tensor,
        flow_steps: int,
    ) -> torch.Tensor:
        """Integrate with the explicit midpoint method using two actor calls per step."""
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
            velocity = self.actor(torch.cat([observations, embedded_t, x_t], dim=-1))
            velocity = self.mlp_output_scale * velocity

            x_mid = x_t + 0.5 * dt[i] * velocity
            t_mid = t_val + 0.5 * dt[i]
            scaled_t_mid = t_mid * freqs
            embedded_t_mid = torch.cat(
                [torch.cos(scaled_t_mid), torch.sin(scaled_t_mid)], dim=-1
            )
            embedded_t_mid = embedded_t_mid.expand(batch_size, -1)
            velocity_mid = self.actor(
                torch.cat([observations, embedded_t_mid, x_mid], dim=-1)
            )
            velocity_mid = self.mlp_output_scale * velocity_mid
            x_t = x_t + dt[i] * velocity_mid

        return x_t

    def _integrate_selected_flow(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        dt: torch.Tensor,
        flow_steps: int,
        integration_method: str,
    ) -> torch.Tensor:
        if integration_method == "euler":
            return self._compiled_integrate_flow(
                observations, x_t, t_current, dt, flow_steps
            )
        if integration_method == "midpoint":
            return self._compiled_integrate_flow_midpoint(
                observations, x_t, t_current, dt, flow_steps
            )
        raise ValueError(f"Unknown integration method: {integration_method}")

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
            return torch.sum(squared_errors, dim=-1) / (squared_errors.shape[-1] ** 0.5)

    def act_inference(
        self,
        observations,
        eval_mode="zero",
        eval_fixed_seed=12345,
        source=None,
        integration_method=None,
        sampling_steps=None,
    ):
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

        if source is not None:
            if source.shape != (batch_size, self.num_actions):
                raise ValueError(
                    f"source must have shape {(batch_size, self.num_actions)}, got {source.shape}"
                )
            x_t = source.to(device=device, dtype=observations.dtype)
        elif eval_mode == "zero":
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

        flow_steps = self.sampling_steps if sampling_steps is None else sampling_steps
        method = (
            self.integration_method
            if integration_method is None
            else integration_method
        )
        full_t_path = torch.linspace(1.0, 0.0, flow_steps + 1, device=device)
        t_current = full_t_path[:-1]
        t_next = full_t_path[1:]
        dt = t_next - t_current

        x_t = self._integrate_selected_flow(
            observations, x_t, t_current, dt, flow_steps, method
        )

        actions = self.actor_scale * x_t
        return actions

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
