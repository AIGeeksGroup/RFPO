#!/usr/bin/env python

"""Flow Matching Policy Implementation"""
from collections import deque
from typing import Union

import torch
import torch.nn.functional as F
import torchvision
from diffusers import EMAModel
from lerobot.common.constants import ACTION, OBS_IMAGES
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from torch import Tensor, nn

from src.antithetic_inference import average_antithetic_predictions, build_antithetic_sources
from src.rollout_bookkeeping import apply_source_sampling_scale, apply_zero_sampling_mask

from .flow_model_config import FlowMatchingConfig
from .flow_integrators import explicit_midpoint_step
from .cfm_predictions import flow_endpoint_predictions
from .flow_net_mlp import FlowMatchingMLPModel
from .flow_net_unet import FlowMatchingUnetModel
from .flow_net_residual_mlp import FlowMatchingResidualMLPModel
from .noise_injection_network import NoiseInjectionNetwork
from .source_priors import apply_previous_action_prior, update_ar1_gaussian_source


# Helper functions for vision encoder
def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    """Get a ResNet model with the final FC layer removed."""
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = nn.Identity()
    return resnet


def replace_bn_with_gn(module: nn.Module, features_per_group: int = 16) -> nn.Module:
    """Replace all BatchNorm layers with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_groups = child.num_features // features_per_group
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            replace_bn_with_gn(child, features_per_group)
    return module

class FlowMatchingPolicy(PreTrainedPolicy):
    """
    Flow Matching Policy implementation for behavior cloning.
    Uses continuous normalizing flows with flow matching loss.
    """

    config_class = FlowMatchingConfig
    name = "flowmatching"

    def __init__(
        self,
        config: FlowMatchingConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Create feature dictionaries for normalization
        input_feature_dict = {}
        for feat in config.input_features:
            # Determine feature type based on name
            if "image" in feat:
                feat_type = FeatureType.VISUAL
                shape = config.input_shapes.get(feat, (3, 96, 96))
                # Convert HWC to CHW format if needed
                if len(shape) == 3 and shape[2] <= 4:  # Likely HWC format (e.g., 84, 84, 3)
                    shape = (shape[2], shape[0], shape[1])  # Convert to CHW
            elif "state" in feat or "pos" in feat:
                feat_type = FeatureType.STATE
                shape = config.input_shapes.get(feat, (2,))
            else:
                feat_type = FeatureType.ENV
                shape = config.input_shapes.get(feat, (1,))

            input_feature_dict[feat] = PolicyFeature(type=feat_type, shape=shape)

        output_feature_dict = {}
        for feat in config.output_features:
            if feat == "action":
                feat_type = FeatureType.ACTION
                shape = config.output_shapes.get(feat, (2,))
            else:
                feat_type = FeatureType.ACTION
                shape = config.output_shapes.get(feat, (1,))

            output_feature_dict[feat] = PolicyFeature(type=feat_type, shape=shape)

        # Normalization layers
        self.normalize_inputs = Normalize(input_feature_dict, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(output_feature_dict, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(output_feature_dict, config.normalization_mapping, dataset_stats)

        # Initialize the flow model based on network architecture
        if config.network_architecture == "mlp":
            self.model = FlowMatchingMLPModel(config)
        elif config.network_architecture == "residual_mlp":
            self.model = FlowMatchingResidualMLPModel(config)
        elif config.network_architecture == "unet":
            self.model = FlowMatchingUnetModel(config)
        else:
            raise ValueError(f"Unknown network_architecture: {config.network_architecture}")

        # Initialize EMA model if ema_power > 0
        self.ema_model = None
        if config.ema_power > 0:
            self.ema_model = EMAModel(
                self.model.parameters(),
                power=config.ema_power,
                model_cls=type(self.model),
                model_config=None,
            )

        # Exploration noise value (can be set dynamically)
        self.exploration_noise_std = config.exploration_noise_std  

        self.num_envs = None
        self.action_buffers = None

        self.reset()

    def init_action_buffers(self, num_envs: int):
        # Initialize empty action buffers dict 
        self.num_envs = num_envs
        self.action_buffers = {
            env_id: deque([], maxlen=self.config.n_action_steps) for env_id in range(self.num_envs)
        }
        self.previous_action_buffers = {
            env_id: deque([], maxlen=self.config.n_action_steps) for env_id in range(self.num_envs)
        }
        self.temporal_source_buffers = {env_id: None for env_id in range(self.num_envs)}
        self.mdp_x_t_path_buffers = {
            env_id: deque([], maxlen=self.config.n_action_steps) for env_id in range(self.num_envs)
        }
            
    def get_optim_params(self) -> dict:
        """Get optimizer parameters with different learning rates for backbone and other parts."""
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.vision_encoder") and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if n.startswith("model.vision_encoder") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self, env_ids: Tensor | None = None):
        """Reset the action buffers for specified environments or all if env_ids is None.

        Args:
            env_ids: Optional tensor of environment indices to reset. If None, resets all buffers.
        """
        if env_ids is None:
            if self.num_envs is None:
                self.action_buffers = {}
                self.mdp_x_t_path_buffers = {}
                self.previous_action_buffers = {}
                self.temporal_source_buffers = {}

            else:
                # Reset all buffers
                for env_id in range(self.num_envs):
                    self.action_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
                    self.mdp_x_t_path_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
                    self.previous_action_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
                    self.temporal_source_buffers[env_id] = None
        else:
            # Reset only specified environment buffers
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids)

            for env_id in env_ids.tolist():
                if env_id in self.action_buffers:
                    self.action_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
                    self.mdp_x_t_path_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
                    self.previous_action_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
                    self.temporal_source_buffers[env_id] = None
    def step_ema(self):
        """Update the EMA model with current model parameters."""
        if self.ema_model is not None:
            self.ema_model.step(self.model.parameters())

    def enable_ema(self):
        """Switch to using EMA weights for inference."""
        if self.ema_model is not None:
            # Store current parameters before copying EMA weights
            self.ema_model.store(self.model.parameters())
            # Copy EMA weights to model
            self.ema_model.copy_to(self.model.parameters())

    def disable_ema(self):
        """Restore original weights after using EMA."""
        if self.ema_model is not None:
            # Restore the parameters that were stored in enable_ema()
            self.ema_model.restore(self.model.parameters())

    def get_schedule(self, device: torch.device) -> Tensor:
        """Get the schedule for the flow matching."""
        return torch.linspace(1.0, 0.0, self.config.sampling_steps + 1, device=device)

    @torch.no_grad
    def select_action(
        self,
        batch: dict[str, Tensor],
        zero_sampling: bool = False,
        sde_sampling: bool = False,
        antithetic_sampling: bool = False,
        zero_sampling_mask: Tensor | None = None,
        source_sampling_scale: Tensor | None = None,
        temporal_source_correlation: float = 0.0,
    ) -> Tensor:
        """Select actions for multiple environments with separate buffers.
        
        Args:
            batch: Dictionary of observations with tensors of shape (num_envs, ...)
            
        Returns:
            Tensor of actions with shape (num_envs, action_dim)
        """
        self.eval()
        
        if self.num_envs is None and self.action_buffers is None:
            raise ValueError("Action buffers not initialized. Call init_action_buffers first.")
        if antithetic_sampling:
            if zero_sampling or sde_sampling or zero_sampling_mask is not None or source_sampling_scale is not None:
                raise ValueError("antithetic sampling cannot be combined with other source or SDE options")
            if self.config.source_prior_mode != "gaussian":
                raise ValueError("antithetic sampling requires the Gaussian source prior")
        if not 0.0 <= temporal_source_correlation < 1.0:
            raise ValueError("temporal_source_correlation must be in [0, 1)")
        if temporal_source_correlation > 0:
            if (
                zero_sampling
                or sde_sampling
                or antithetic_sampling
                or zero_sampling_mask is not None
                or source_sampling_scale is not None
            ):
                raise ValueError(
                    "temporal source correlation cannot be combined with other source or SDE options"
                )
            if self.config.source_prior_mode != "gaussian":
                raise ValueError(
                    "temporal source correlation requires the Gaussian source prior"
                )
        if zero_sampling_mask is not None:
            if zero_sampling or source_sampling_scale is not None:
                raise ValueError("zero sampling options and source_sampling_scale cannot be combined")
            if zero_sampling_mask.shape != (self.num_envs,):
                raise ValueError(
                    f"zero_sampling_mask must have shape ({self.num_envs},), "
                    f"got {tuple(zero_sampling_mask.shape)}"
                )
        if source_sampling_scale is not None and source_sampling_scale.shape != (self.num_envs,):
            raise ValueError(
                f"source_sampling_scale must have shape ({self.num_envs},), "
                f"got {tuple(source_sampling_scale.shape)}"
            )
        
        # Check which environments need new action chunks
        envs_needing_actions = []
        for env_id in range(self.num_envs):
            if len(self.action_buffers[env_id]) == 0:
                envs_needing_actions.append(env_id)
        
        # Generate new action chunks for environments that need them
        if envs_needing_actions:
            # Create batch for environments needing actions
            sub_batch = {}
            for key, value in batch.items():
                sub_batch[key] = value[envs_needing_actions]

            previous_actions = None
            has_previous_actions = None
            if self.config.source_prior_mode == "previous_action":
                action_dim = self.model.action_dim
                previous_actions = torch.zeros(
                    (len(envs_needing_actions), self.config.n_action_steps, action_dim),
                    device=next(iter(sub_batch.values())).device,
                )
                has_previous_actions = torch.zeros(
                    len(envs_needing_actions), dtype=torch.bool, device=previous_actions.device
                )
                for i, env_id in enumerate(envs_needing_actions):
                    history = self.previous_action_buffers[env_id]
                    if len(history) == self.config.n_action_steps:
                        previous_actions[i] = torch.stack(list(history))
                        has_previous_actions[i] = True

            if antithetic_sampling:
                source_noise = torch.randn(
                    len(envs_needing_actions),
                    self.config.horizon,
                    self.model.action_dim,
                    device=next(iter(sub_batch.values())).device,
                )
                positive_source, negative_source = build_antithetic_sources(source_noise)
                positive_actions, positive_path = self.predict_action_chunk(
                    dict(sub_batch), source_noise=positive_source
                )
                negative_actions, negative_path = self.predict_action_chunk(
                    dict(sub_batch), source_noise=negative_source
                )
                action_chunks = average_antithetic_predictions(positive_actions, negative_actions)
                mdp_x_t_path = average_antithetic_predictions(positive_path, negative_path)
            elif temporal_source_correlation > 0:
                innovations = torch.randn(
                    len(envs_needing_actions),
                    self.config.horizon,
                    self.model.action_dim,
                    device=next(iter(sub_batch.values())).device,
                )
                correlated_sources = []
                for i, env_id in enumerate(envs_needing_actions):
                    source = update_ar1_gaussian_source(
                        innovations[i],
                        self.temporal_source_buffers[env_id],
                        temporal_source_correlation,
                    )
                    self.temporal_source_buffers[env_id] = source.clone()
                    correlated_sources.append(source)
                action_chunks, mdp_x_t_path = self.predict_action_chunk(
                    sub_batch, source_noise=torch.stack(correlated_sources)
                )
            else:
                action_chunks, mdp_x_t_path = self.predict_action_chunk(
                    sub_batch,
                    zero_sampling=zero_sampling,
                    sde_sampling=sde_sampling,
                    zero_sampling_mask=(
                        zero_sampling_mask[envs_needing_actions]
                        if zero_sampling_mask is not None
                        else None
                    ),
                    source_sampling_scale=(
                        source_sampling_scale[envs_needing_actions]
                        if source_sampling_scale is not None
                        else None
                    ),
                    previous_actions=previous_actions,
                    has_previous_actions=has_previous_actions,
                )
            action_chunks = action_chunks[:, :self.config.n_action_steps, :] # in future, the n_action_steps can be different from prediction horizon
            mdp_x_t_path = mdp_x_t_path[:, :, :self.config.n_action_steps, :] # in future, the n_action_steps can be different from prediction horizon
            assert mdp_x_t_path.shape[1] == self.config.sampling_steps, "mdp_x_t_path second axis should be the flow sampling steps"
            assert action_chunks.shape[1] == self.config.n_action_steps, "action_chunks second axis should be the number of action steps"

            # Fill the buffers for these environments
            for i, env_id in enumerate(envs_needing_actions):
                # Transpose to get actions for this environment across timesteps
                self.action_buffers[env_id].extend(action_chunks[i])
                self.mdp_x_t_path_buffers[env_id].extend(mdp_x_t_path[i].transpose(0,1))

        # Collect actions for all environments
        actions = []
        mdp_x_t_paths = []
        for env_id in range(self.num_envs):
            action = self.action_buffers[env_id].popleft()
            actions.append(action)
            self.previous_action_buffers[env_id].append(action)
            mdp_x_t_paths.append(self.mdp_x_t_path_buffers[env_id].popleft())

        assert len(actions) == self.num_envs, "actions length should be the number of environments"
        assert len(mdp_x_t_paths) == self.num_envs, "mdp_x_t_paths length should be the number of environments"
        
        return torch.stack(actions), torch.stack(mdp_x_t_paths)

    @torch.no_grad
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        zero_sampling: bool = False,
        sde_sampling: bool = False,
        previous_actions: Tensor | None = None,
        has_previous_actions: Tensor | None = None,
        source_noise: Tensor | None = None,
        zero_sampling_mask: Tensor | None = None,
        source_sampling_scale: Tensor | None = None,
    ) -> Tensor:
        """Predict a chunk of actions by integrating the learned flow field."""
        self.eval()

        if sde_sampling and self.config.integration_method != "euler":
            raise ValueError("sde_sampling is only supported with Euler integration")

        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Get observation conditioning
        obs_cond = self.model.encode_observations(batch)

        # Initialize from the configured source distribution.
        B = obs_cond.shape[0]
        expected_source_shape = (B, self.config.horizon, self.model.action_dim)
        if source_noise is not None:
            if zero_sampling or zero_sampling_mask is not None or source_sampling_scale is not None:
                raise ValueError("source_noise cannot be combined with zero sampling options")
            if source_noise.shape != expected_source_shape:
                raise ValueError(
                    f"source_noise must have shape {expected_source_shape}, got {tuple(source_noise.shape)}"
                )
            x_t = source_noise.to(device=obs_cond.device).clone()
        elif zero_sampling:
            x_t = torch.zeros((B, self.config.horizon, self.model.action_dim), device=obs_cond.device)
        else:
            x_t = torch.randn((B, self.config.horizon, self.model.action_dim), device=obs_cond.device)

        if zero_sampling_mask is not None:
            x_t = apply_zero_sampling_mask(x_t, zero_sampling_mask)
        if source_sampling_scale is not None:
            x_t = apply_source_sampling_scale(x_t, source_sampling_scale)

        if self.config.source_prior_mode == "previous_action":
            if previous_actions is None or has_previous_actions is None:
                raise ValueError("previous_action source prior requires action history and its availability mask")
            normalized_previous_actions = self.normalize_targets({ACTION: previous_actions})[ACTION]
            x_t = apply_previous_action_prior(
                source_noise=x_t,
                previous_actions=normalized_previous_actions,
                has_previous_actions=has_previous_actions,
                sigma=self.config.source_prior_sigma,
            )

        # Flow schedule for numerical integration
        flow_steps = self.config.sampling_steps
        # t_path = torch.linspace(1.0, 0.0, flow_steps + 1, device=x_t.device)
        t_path = self.get_schedule(x_t.device)
        mdp_x_t_path = torch.zeros((B, flow_steps, self.config.horizon, self.model.action_dim), device=x_t.device)

        if sde_sampling:
            sde_sigma = self.get_sde_sigma(obs_cond, x_t.device)
            if sde_sigma.shape == (B, self.config.horizon, 1, self.model.action_dim):
                sde_sigma = sde_sigma.expand(B, self.config.horizon, flow_steps, self.model.action_dim)
            elif sde_sigma.shape == (flow_steps,):
                sde_sigma = sde_sigma.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, self.config.horizon, flow_steps, self.model.action_dim)
            else:
                raise ValueError(f"sde_sigma shape should be (B, horizon, 1, D) or (flow_steps,), but got {sde_sigma.shape}")

        def velocity_at(state: Tensor, time: Tensor) -> Tensor:
            timeembedding = self.model.diffusion_step_encoder(time.reshape(1))
            assert timeembedding.shape == (1, self.config.timestep_embed_dim)
            timeembedding = timeembedding.expand(B, -1)

            network_output = self.config.mlp_output_scale * self.model(state, timeembedding, obs_cond)
            assert network_output.shape == (
                B,
                self.config.horizon,
                self.model.action_dim,
            ), "network_output shape should be (B, horizon, action_dim)"
            if self.config.transported_clip_value is not None:
                network_output = network_output.clamp(
                    -self.config.transported_clip_value,
                    self.config.transported_clip_value,
                )

            if self.config.flow_network_output_param == "u":
                return network_output
            if time.item() > 1e-5:
                return (state - network_output) / time
            return torch.zeros_like(state)

        # Integrate from t=1 to t=0.
        for i in range(flow_steps):
            t_current = t_path[i]
            t_next = t_path[i + 1]
            dt = t_next - t_current  # Negative value since we go from 1 to 0

            if self.config.integration_method == "euler":
                x_t = x_t + velocity_at(x_t, t_current) * dt
            else:
                x_t = explicit_midpoint_step(x_t, t_current, dt, velocity_at)
            if sde_sampling:
                # x_t = x_t + self.config.sde_sigma * torch.randn_like(x_t)
                x_t = x_t + sde_sigma[:, :, i, :] * torch.randn_like(x_t)
            mdp_x_t_path[:, i] = x_t

        # Scale actions
        actions = self.config.actor_scale * x_t
        
        # Add exploration noise if configured and in training mode
        if self.training and self.exploration_noise_std > 0:
            noise = self.exploration_noise_std * torch.randn_like(actions)
            actions = actions + noise

        # Unnormalize actions
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        return actions, mdp_x_t_path

    def forward(self, batch: dict[str, Tensor], n_action_samples: int = 1, cfm_loss_t: Tensor = None, cfm_loss_eps: Tensor = None, debug=False, is_dppo: bool = False, return_cfm_predictions: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        if is_dppo:
            return self.forward_dppo(batch, debug)
        else:
            return self.forward_fpo(
                batch,
                n_action_samples,
                cfm_loss_t,
                cfm_loss_eps,
                debug,
                return_cfm_predictions=return_cfm_predictions,
            )

    def forward_dppo(self, batch: dict[str, Tensor], debug=False) -> tuple[Tensor, Tensor, Tensor]:
        # Compute denoising likelihood
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Get observation conditioning
        obs_cond = self.model.encode_observations(batch)

        # Concatenate the mdp_x_t_path and the action (x0)
        x0 = batch[ACTION] 
        mdp_x_t_path = batch["mdp_x_t_path"] # start from x1
        B, T, D = x0.shape
        assert mdp_x_t_path.shape == (B, T, self.config.sampling_steps, D), "mdp_x_t_path shape should be (B, T, flow sampling_steps, D)"

        full_x_t_path = torch.cat([mdp_x_t_path, x0.unsqueeze(2)], dim=2)
        assert full_x_t_path.shape == (B, T, self.config.sampling_steps + 1, D), "full_x_t_path shape should be (B, T, flow sampling_steps + 1, D)"

        # Compute the denoising likelihood
        x_t = full_x_t_path[..., :-1, :]
        x_t_next = full_x_t_path[..., 1:, :]
        assert x_t.shape == (B, T, self.config.sampling_steps, D), "x_t shape should be (B, T, flow sampling_steps, D)"
        assert x_t_next.shape == (B, T, self.config.sampling_steps, D), "x_t_next shape should be (B, T, flow sampling_steps, D)"

        t_path = self.get_schedule(x_t.device) 
        assert t_path.shape == (self.config.sampling_steps + 1,), "t_path shape should be (flow sampling_steps + 1,)"

        flow_steps = self.config.sampling_steps
        expected_x_t_next = torch.zeros((B, T, self.config.sampling_steps, D), device=x_t.device)
        for i in range(flow_steps):
            t_current = t_path[i]
            t_next = t_path[i + 1]
            dt = t_next - t_current  # Negative value since we go from 1 to 0
            
            timeembedding = self.model.diffusion_step_encoder(torch.tensor([t_current], device=x_t.device))
            assert timeembedding.shape == (1, self.config.timestep_embed_dim)
            timeembedding = timeembedding.expand(B, -1)

            # Predict based on output parameterization
            network_output = self.model(x_t[:, :, i, :], timeembedding, obs_cond)
            network_output = self.config.mlp_output_scale * network_output

            # Apply clipping if configured
            if self.config.transported_clip_value is not None:
                network_output = network_output.clamp(-self.config.transported_clip_value, self.config.transported_clip_value)
            
            if self.config.flow_network_output_param == "u":
                # Direct velocity prediction
                velocity = network_output
            elif self.config.flow_network_output_param == "x0":
                # x0 prediction mode: compute velocity from x0
                x0_pred = network_output

                # Velocity: u = (x_t - x0) / t for t > 0
                if t_current > 1e-5:  # Avoid division by zero
                    velocity = (x_t[:, :, i, :] - x0_pred) / t_current
                else:
                    velocity = torch.zeros_like(x_t)
            
            # Euler step: x_{t+dt} = x_t + velocity * dt
            expected_x_t_next[:, :, i, :] = x_t[:, :, i, :] + dt * velocity

        realized_noise = x_t_next - expected_x_t_next
        assert realized_noise.shape == (B, T, self.config.sampling_steps, D), "realized_noise shape should be (B, T, flow sampling_steps, D)"

        # standardize the realized noise and compute log probability
        sde_sigma = self.get_sde_sigma(obs_cond, x_t.device)

        if sde_sigma.shape == (self.config.sampling_steps,):
            # Fixed sigma per flow step: shape (sampling_steps,)
            # Broadcast to (1, 1, sampling_steps, 1) for division
            sigma_expanded = sde_sigma[None, None, :, None]
            standardized_noise = realized_noise / (sigma_expanded + 1e-6)

            # Log probability: -0.5 * ||z||^2 - D/2 * log(2pi) - D * log(sigma)
            # The last term accounts for the Jacobian of the transformation
            log_prob = (
                -0.5 * torch.sum(standardized_noise**2, dim=-1)
                - D / 2 * torch.log(torch.tensor(2 * torch.pi, device=x_t.device))
                - D * torch.log(sde_sigma + 1e-6)[None, None, :]
            )
            # Entropy is 0 for fixed sigma (not learnable)
            entropy = torch.tensor(0.0, device=x_t.device)

        elif sde_sigma.shape == (B, T, 1, D):
            # Learned sigma: shape (B, T, 1, D)
            # Broadcasts across sampling_steps dimension
            standardized_noise = realized_noise / (sde_sigma + 1e-6)

            # Log probability: -0.5 * ||z||^2 - D/2 * log(2pi) - sum_d(log(sigma_d))
            # Sum over action dimension D, the sigma is same across sampling_steps
            log_sigma_sum = torch.sum(torch.log(sde_sigma + 1e-6), dim=-1)  # (B, T, 1)
            log_prob = (
                -0.5 * torch.sum(standardized_noise**2, dim=-1)
                - D / 2 * torch.log(torch.tensor(2 * torch.pi, device=x_t.device))
                - log_sigma_sum.squeeze(-1).unsqueeze(-1).expand(-1, -1, self.config.sampling_steps)
            )

            # Entropy of Gaussian: 0.5 * D * (1 + log(2*pi)) + sum_d(log(sigma_d))
            # Shape: (B, T, 1) -> squeeze to (B, T)
            entropy = (
                0.5 * D * (1 + torch.log(torch.tensor(2 * torch.pi, device=x_t.device)))
                + log_sigma_sum.squeeze(-1)
            )
        else:
            raise ValueError(
                f"sde_sigma shape should be (sampling_steps,) or (B, T, 1, D), "
                f"but got {sde_sigma.shape}"
            )

        assert log_prob.shape == (B, T, self.config.sampling_steps), \
            f"log_prob shape should be (B, T, flow sampling_steps), but got {log_prob.shape}"

        return log_prob, entropy, sde_sigma

    def get_sde_sigma(self, obs_cond: Tensor, device: torch.device) -> Tensor:
        """Get the SDE sigma for the flow matching.

        Args:
            obs_cond: Observation conditioning tensor of shape (B, obs_cond_dim)
            device: Device to place the tensor on

        Returns:
            If learn_sde_sigma=False: Tensor of shape (sampling_steps,) with fixed sigma
            If learn_sde_sigma=True: Tensor of shape (B, T, 1, D) from noise injection network
                where T is horizon and D is action_dim. The singleton dimension broadcasts
                across flow sampling steps.
        """
        if getattr(self.config, 'learn_sde_sigma', False):
            if getattr(self, 'noise_injection_network', None) is None:
                raise RuntimeError("Noise injection network not initialized. Call initialize_noise_injection_network() first.")
            # obs_cond shape: (B, obs_cond_dim)
            # Output shape: (B, T, 1, D) where T=horizon, D=action_dim
            sde_sigma = self.noise_injection_network(obs_cond)
            return sde_sigma
        else:
            return torch.ones((self.config.sampling_steps,), device=device) * self.config.sde_sigma

    def initialize_noise_injection_network(self):
        """Initialize the noise injection network.

        Creates a NoiseInjectionNetwork that takes obs_cond as input and outputs
        a tensor with shape (B, T, 1, D) where:
            - B: batch size
            - T: action horizon (chunk size)
            - 1: singleton dimension for broadcasting across flow sampling steps
            - D: action dimension
        """
        # Get observation conditioning dimension from the model
        obs_cond_dim = self.model.global_cond_dim
        action_dim = self.model.action_dim
        horizon = self.config.horizon

        min_noise = getattr(self.config, 'noise_injection_min', 0.2)
        max_noise = getattr(self.config, 'noise_injection_max', 0.5)

        self.noise_injection_network = NoiseInjectionNetwork(
            obs_cond_dim=obs_cond_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dims=[256, 256],
            min_noise=min_noise,
            max_noise=max_noise,
        )

    def forward_fpo(self, batch: dict[str, Tensor], n_action_samples: int = 1, cfm_loss_t: Tensor = None, cfm_loss_eps: Tensor = None, debug=False, return_cfm_predictions: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        actions = batch[ACTION] # (num_envs, horizon, action_dim)
        B, T, D = actions.shape
        
        if cfm_loss_t is None and cfm_loss_eps is None:
            # Sample random timesteps uniformly from [0, 1]
            cfm_loss_t = torch.rand((B * n_action_samples, 1, 1), device=actions.device)
            # Sample noise (x1)
            cfm_loss_eps = torch.randn((B * n_action_samples, T, D), device=actions.device)

        if n_action_samples > 1:
            # Copy the observations n_action_samples times
            if self.config.image_features:
                for img_key in self.config.image_features:
                    if img_key in batch:
                        img_tensor = batch[img_key]
                        img_tensor_shape = img_tensor.shape
                        if len(img_tensor_shape) == 5:
                            img_tensor = img_tensor.unsqueeze(1).expand(B, n_action_samples, -1, -1, -1, -1)
                            batch[img_key] = img_tensor.reshape(-1, *img_tensor_shape[1:])
                        else:
                            img_tensor = img_tensor.unsqueeze(1).expand(B, n_action_samples, -1, -1, -1)
                            batch[img_key] = img_tensor.reshape(-1, *img_tensor_shape[1:])
            if self.config.state_features:
                for state_key in self.config.state_features:
                    if state_key in batch:
                        state_tensor = batch[state_key]
                        state_tensor_shape = state_tensor.shape
                        if len(state_tensor_shape) == 2:
                            state_tensor = state_tensor.unsqueeze(1).expand(B, n_action_samples, -1)
                            batch[state_key] = state_tensor.reshape(-1, *state_tensor_shape[1:])
                        else:
                            state_tensor = state_tensor.unsqueeze(1).expand(B, n_action_samples, -1, -1)
                            batch[state_key] = state_tensor.reshape(-1, *state_tensor_shape[1:])

            # Reshape actions in to (B * n_action_samples, T, D)
            actions = actions.unsqueeze(1).expand(-1, n_action_samples, -1, -1).reshape(-1, T, D)
            batch[ACTION] = actions

        cfm_output = self.get_cfm_loss(
            batch,
            cfm_loss_eps,
            cfm_loss_t,
            non_reduction=True,
            return_predictions=return_cfm_predictions,
        )
        if return_cfm_predictions:
            cfm_loss, x1_pred, _ = cfm_output
        else:
            cfm_loss = cfm_output

        # Reshape the cfm_loss, cfm_loss_t, cfm_loss_eps
        cfm_loss = cfm_loss.mean(-1).transpose(0,1).reshape(T, B, n_action_samples) # (B * n_action_samples, T) -> (T,B*n_action_samples) -> (T,B,n_action_samples)
        cfm_loss_t = cfm_loss_t.squeeze(-1).transpose(0,1).expand(T, -1).reshape(T, B, n_action_samples) # (B*n_action_samples, 1, 1) -> (T, B, n_action_samples)
        cfm_loss_eps = cfm_loss_eps.permute(1, 0, 2).reshape(T, B, n_action_samples, D) # (B*n_action_samples, T, D) -> (T, B, n_action_samples, D)

        if return_cfm_predictions:
            x1_pred = x1_pred.permute(1, 0, 2).reshape(T, B, n_action_samples, D)
            return cfm_loss, cfm_loss_t, cfm_loss_eps, x1_pred
        return cfm_loss, cfm_loss_t, cfm_loss_eps

    def get_cfm_loss(self, batch: dict[str, Tensor], cfm_loss_eps: Tensor = None, cfm_loss_t: Tensor = None, non_reduction: bool = False, return_predictions: bool = False) -> tuple[Tensor, dict]:
        """Forward pass for training with flow matching loss."""
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Get observation conditioning
        obs_cond = self.model.encode_observations(batch)

        # Get clean actions (x0)
        actions = batch[ACTION]  # Shape: (B, T, D)
        B, T, D = actions.shape

        normalized_previous_actions = None
        has_previous_actions = batch.get("has_previous_actions")
        if self.config.source_prior_mode == "previous_action" and "previous_action" in batch:
            normalized_previous_actions = self.normalize_targets(
                {ACTION: batch["previous_action"]}
            )[ACTION]

        if cfm_loss_eps is None or cfm_loss_t is None:
            # Sample random timesteps uniformly from [0, 1]
            t = torch.rand((B, 1, 1), device=actions.device)
            # Sample noise (x1)
            noise = torch.randn_like(actions)
            if normalized_previous_actions is not None and has_previous_actions is not None:
                noise = apply_previous_action_prior(
                    source_noise=noise,
                    previous_actions=normalized_previous_actions,
                    has_previous_actions=has_previous_actions,
                    sigma=self.config.source_prior_sigma,
                )
        else:
            t = cfm_loss_t
            noise = cfm_loss_eps
        
        # Interpolate between x0 and x1: x_t = (1-t) * x0 + t * x1
        x_t = (1 - t) * actions + t * noise
        
        # Predict based on output parameterization
        timeembedding = self.model.diffusion_step_encoder(t)
        timeembedding = timeembedding.reshape(B, -1)
        network_output = self.model(x_t, timeembedding, obs_cond)
        network_output = self.config.mlp_output_scale * network_output

        # Apply clipping if configured
        if self.config.transported_clip_value is not None:
            network_output = network_output.clamp(-self.config.transported_clip_value, self.config.transported_clip_value)

        velocity_pred, x0_pred, x1_pred = flow_endpoint_predictions(
            x_t,
            t,
            network_output,
            output_parameterization=self.config.flow_network_output_param,
        )

        # Compute loss based on mode
        if self.config.cfm_loss_mode == "x0":
            loss = self._compute_squared_error(x0_pred, actions)
            
        elif self.config.cfm_loss_mode == "u":
            # Velocity MSE loss
            target_velocity = noise - actions  # True flow velocity from x0 to x1
            loss = self._compute_squared_error(velocity_pred, target_velocity)
            
        elif self.config.cfm_loss_mode == "eps":
            loss = self._compute_squared_error(x1_pred, noise)
        
        # Apply timestep-dependent weighting
        weight = self._compute_cfm_loss_weight(t)
        loss = loss * weight
        
        if non_reduction:
            if return_predictions:
                return loss, x1_pred, x0_pred
            return loss
        
        # Handle padding mask if present
        if "action_is_pad" in batch:
            mask = ~batch["action_is_pad"].unsqueeze(-1)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        loss_dict = {"flow_loss": loss.item()}
        return loss, loss_dict

    def _compute_cfm_loss_weight(self, t: Tensor) -> Tensor:
        """Compute timestep-dependent weight for CFM loss."""
        # t has shape (B, 1, 1)
        t_scalar = t.squeeze(-1).squeeze(-1)  # Shape: (B,)
        
        if self.config.cfm_loss_weight_from_t == "constant":
            weight = torch.ones_like(t)
        elif self.config.cfm_loss_weight_from_t == "linear_1_to_0.1":
            # Weight goes from 1.0 at t=0 to 0.1 at t=1
            weight = 1.0 - 0.9 * t
        elif self.config.cfm_loss_weight_from_t == "linear_1_to_0.01":
            # Weight goes from 1.0 at t=0 to 0.01 at t=1
            weight = 1.0 - 0.99 * t
        
        return weight

    def _compute_squared_error(
        self, predictions: Tensor, targets: Tensor
    ) -> torch.Tensor:
        """Compute squared error with optional Huber loss."""
        if self.config.cfm_loss_use_huber:
            # Modified Huber loss to match MSE when |error| <= delta
            # L2 for |error| <= delta, linear for |error| > delta
            diff = predictions - targets
            abs_diff = torch.abs(diff)
            huber_loss = torch.where(
                abs_diff <= self.config.cfm_loss_huber_delta,
                diff**2,  # No 0.5 factor, matches MSE
                2 * self.config.cfm_loss_huber_delta * abs_diff - self.config.cfm_loss_huber_delta**2,
            )
            return huber_loss
        else:
            return F.mse_loss(predictions, targets, reduction="none")
