#!/usr/bin/env python

"""Flow Matching Policy Implementation"""
import copy
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

from .flow_model_config import FlowMatchingConfig
from .flow_net_mlp import FlowMatchingMLPModel
from .flow_net_unet import FlowMatchingUnetModel
from .flow_net_residual_mlp import FlowMatchingResidualMLPModel
from .noise_injection_network import NoiseInjectionNetwork
from .reflow_extensions import (
    StepPredictor,
    compute_advantage_reflow_weights,
    compute_discretization_gap,
    compute_path_straightness,
)


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

        # Detached flow copy for fpo_operator reflow endpoints (never swap training weights).
        self._reflow_endpoint_model = None
        self._ema_update_count = 0
        # Frozen teacher flow net (not registered as a submodule, so checkpoints stay student-only).
        self.__dict__["_teacher_model"] = None

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

            else:
                # Reset all buffers
                for env_id in range(self.num_envs):
                    self.action_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
                    self.mdp_x_t_path_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
        else:
            # Reset only specified environment buffers
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids)

            for env_id in env_ids.tolist():
                if env_id in self.action_buffers:
                    self.action_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
                    self.mdp_x_t_path_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
    def step_ema(self):
        """Update the EMA model with current model parameters."""
        if self.ema_model is not None:
            self.ema_model.step(self.model.parameters())
            self._ema_update_count += 1

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
    def select_action(self, batch: dict[str, Tensor], zero_sampling: bool = False, sde_sampling: bool = False) -> Tensor:
        """Select actions for multiple environments with separate buffers.
        
        Args:
            batch: Dictionary of observations with tensors of shape (num_envs, ...)
            
        Returns:
            Tensor of actions with shape (num_envs, action_dim)
        """
        self.eval()
        
        if self.num_envs is None and self.action_buffers is None:
            raise ValueError("Action buffers not initialized. Call init_action_buffers first.")
        
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

            # Predict action chunks for these environments
            action_chunks, mdp_x_t_path = self.predict_action_chunk(sub_batch, zero_sampling=zero_sampling, sde_sampling=sde_sampling)
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
            actions.append(self.action_buffers[env_id].popleft())
            mdp_x_t_paths.append(self.mdp_x_t_path_buffers[env_id].popleft())

        assert len(actions) == self.num_envs, "actions length should be the number of environments"
        assert len(mdp_x_t_paths) == self.num_envs, "mdp_x_t_paths length should be the number of environments"
        
        return torch.stack(actions), torch.stack(mdp_x_t_paths)

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor], zero_sampling: bool = False, sde_sampling: bool = False) -> Tensor:
        """Predict a chunk of actions using flow matching with Euler integration."""
        self.eval()

        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Get observation conditioning
        obs_cond = self.model.encode_observations(batch)

        # Initialize from Gaussian noise
        B = obs_cond.shape[0]
        if zero_sampling:
            x_t = torch.zeros((B, self.config.horizon, self.model.action_dim), device=obs_cond.device)
        else:
            x_t = torch.randn((B, self.config.horizon, self.model.action_dim), device=obs_cond.device)

        # Flow schedule for Euler integration
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

        # Euler integration from t=1 to t=0
        for i in range(flow_steps):
            t_current = t_path[i]
            t_next = t_path[i + 1]
            dt = t_next - t_current  # Negative value since we go from 1 to 0
            
            timeembedding = self.model.diffusion_step_encoder(torch.tensor([t_current], device=x_t.device))
            assert timeembedding.shape == (1, self.config.timestep_embed_dim)
            timeembedding = timeembedding.expand(B, -1)

            # Predict based on output parameterization
            network_output = self.model(x_t, timeembedding, obs_cond)
            network_output = self.config.mlp_output_scale * network_output
            assert network_output.shape == (B, self.config.horizon, self.model.action_dim), "network_output shape should be (B, horizon, action_dim)"

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
                    velocity = (x_t - x0_pred) / t_current
                else:
                    velocity = torch.zeros_like(x_t)
            
            # Euler step: x_{t+dt} = x_t + velocity * dt
            x_t = x_t + velocity * dt 
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

    def forward(self, batch: dict[str, Tensor], n_action_samples: int = 1, cfm_loss_t: Tensor = None, cfm_loss_eps: Tensor = None, debug=False, is_dppo: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        if is_dppo:
            return self.forward_dppo(batch, debug)
        else:
            return self.forward_fpo(batch, n_action_samples, cfm_loss_t, cfm_loss_eps, debug)

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

    def forward_fpo(self, batch: dict[str, Tensor], n_action_samples: int = 1, cfm_loss_t: Tensor = None, cfm_loss_eps: Tensor = None, debug=False) -> tuple[Tensor, Tensor, Tensor]:
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

        cfm_loss = self.get_cfm_loss(batch, cfm_loss_eps, cfm_loss_t, non_reduction=True)

        # Reshape the cfm_loss, cfm_loss_t, cfm_loss_eps
        cfm_loss = cfm_loss.mean(-1).transpose(0,1).reshape(T, B, n_action_samples) # (B * n_action_samples, T) -> (T,B*n_action_samples) -> (T,B,n_action_samples)
        cfm_loss_t = cfm_loss_t.squeeze(-1).transpose(0,1).expand(T, -1).reshape(T, B, n_action_samples) # (B*n_action_samples, 1, 1) -> (T, B, n_action_samples)
        cfm_loss_eps = cfm_loss_eps.permute(1, 0, 2).reshape(T, B, n_action_samples, D) # (B*n_action_samples, T, D) -> (T, B, n_action_samples, D)

        return cfm_loss, cfm_loss_t, cfm_loss_eps

    def get_cfm_loss(self, batch: dict[str, Tensor], cfm_loss_eps: Tensor = None, cfm_loss_t: Tensor = None, non_reduction: bool = False) -> tuple[Tensor, dict]:
        """Forward pass for training with flow matching loss."""
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Get observation conditioning
        obs_cond = self.model.encode_observations(batch)

        # Get clean actions (x0)
        actions = batch[ACTION]  # Shape: (B, T, D)
        B, T, D = actions.shape

        if cfm_loss_eps is None or cfm_loss_t is None:
            # Sample random timesteps uniformly from [0, 1]
            t = torch.rand((B, 1, 1), device=actions.device)
            # Sample noise (x1)
            noise = torch.randn_like(actions)
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

        # Compute loss based on mode
        if self.config.cfm_loss_mode == "x0":
            # x0 MSE loss
            if self.config.flow_network_output_param == "u":
                # If network predicts velocity, compute x0 from it
                velocity_pred = network_output
                x0_pred = x_t - t * velocity_pred
            else:  # flow_network_output_param == "x0"
                x0_pred = network_output
            
            loss = self._compute_squared_error(x0_pred, actions)
            
        elif self.config.cfm_loss_mode == "u":
            # Velocity MSE loss
            target_velocity = noise - actions  # True flow velocity from x0 to x1
            
            if self.config.flow_network_output_param == "u":
                velocity_pred = network_output
            else:  # flow_network_output_param == "x0"
                x0_pred = network_output
                # Compute velocity: u = (x_t - x0) / t
                # Handle t=0 case
                t_clamped = torch.clamp(t, min=1e-5)
                velocity_pred = (x_t - x0_pred) / t_clamped
            
            loss = self._compute_squared_error(velocity_pred, target_velocity)
            
        elif self.config.cfm_loss_mode == "eps":
            # Epsilon (x1) prediction loss
            if self.config.flow_network_output_param == "u":
                velocity_pred = network_output
                x0_pred = x_t - t * velocity_pred
                x1_pred = x0_pred + velocity_pred
            else:  # flow_network_output_param == "x0"
                x0_pred = network_output
                # Compute x1 from x0: x_t = (1-t)*x0 + t*x1 => x1 = (x_t - (1-t)*x0) / t
                t_clamped = torch.clamp(t, min=1e-5)
                x1_pred = (x_t - (1 - t) * x0_pred) / t_clamped
            
            loss = self._compute_squared_error(x1_pred, noise)
        
        # Apply timestep-dependent weighting
        weight = self._compute_cfm_loss_weight(t)
        loss = loss * weight
        
        if non_reduction:
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

    # ------------------------------------------------------------------
    # Reflow idea variants (ported from isaaclab_fpo)
    # ------------------------------------------------------------------
    def ensure_step_predictor(self, step_bins: list[int] | None = None) -> StepPredictor:
        """Lazily create StepPredictor on observation conditioning dim."""
        if getattr(self, "step_predictor", None) is None:
            bins = list(step_bins) if step_bins is not None else [1, 2, 4, 8]
            self.step_predictor = StepPredictor(
                num_obs=self.model.global_cond_dim,
                step_bins=bins,
            )
            # Keep device in sync with the flow network.
            device = next(self.model.parameters()).device
            self.step_predictor.to(device)
        return self.step_predictor

    def _expand_batch_for_samples(self, batch: dict[str, Tensor], n_samples: int) -> dict[str, Tensor]:
        """Repeat observation fields n_samples times along batch dim (no action required)."""
        if n_samples <= 1:
            return batch
        out: dict[str, Tensor] = {}
        for key, value in batch.items():
            if key == ACTION:
                continue
            if not torch.is_tensor(value):
                out[key] = value
                continue
            # (B, ...) -> (B * n_samples, ...)
            B = value.shape[0]
            expanded = value.unsqueeze(1).expand(B, n_samples, *value.shape[1:])
            out[key] = expanded.reshape(B * n_samples, *value.shape[1:])
        return out

    def _ensure_reflow_endpoint_model(self) -> nn.Module:
        """Frozen deepcopy of the flow net for EMA endpoint integration only."""
        if self._reflow_endpoint_model is None:
            endpoint = copy.deepcopy(self.model)
            endpoint.eval()
            for param in endpoint.parameters():
                param.requires_grad_(False)
            self._reflow_endpoint_model = endpoint
        return self._reflow_endpoint_model

    def _sync_reflow_endpoint_from_ema(self) -> None:
        """Copy EMA shadow weights into the detached endpoint model."""
        if self.ema_model is None:
            return
        endpoint = self._ensure_reflow_endpoint_model()
        for param, shadow in zip(endpoint.parameters(), self.ema_model.shadow_params):
            param.data.copy_(shadow.to(param.device))

    def attach_frozen_teacher(self, source_state_dict: dict | None = None) -> None:
        """Freeze a deepcopy of the flow net as teacher for reflow endpoints / KD.

        ``source_state_dict`` may be a full ``FlowMatchingPolicy`` state (keys
        prefixed with ``model.``) or a raw flow-net state. If omitted, the current
        online net is copied (typically the BC / baseline weights at init).
        """
        teacher = copy.deepcopy(self.model)
        if source_state_dict is not None:
            model_sd = {
                key[len("model.") :]: value
                for key, value in source_state_dict.items()
                if key.startswith("model.")
            }
            if model_sd:
                teacher.load_state_dict(model_sd, strict=False)
            else:
                teacher.load_state_dict(source_state_dict, strict=False)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        self.__dict__["_teacher_model"] = teacher

    def _reflow_integration_model(
        self,
        use_ema_endpoint: bool,
        ema_warmup_updates: int = 500,
    ) -> nn.Module | None:
        """Frozen teacher, detached EMA endpoint, or None (online model)."""
        teacher = self.__dict__.get("_teacher_model")
        if teacher is not None:
            return teacher
        if not use_ema_endpoint or self.ema_model is None:
            return None
        if self._ema_update_count < ema_warmup_updates:
            return None
        self._sync_reflow_endpoint_from_ema()
        return self._reflow_endpoint_model

    def _predict_velocity(
        self,
        x_t: Tensor,
        t_scalar: Tensor,
        obs_cond: Tensor,
        flow_model: nn.Module | None = None,
    ) -> Tensor:
        """Predict flow velocity at continuous time t for batch of trajectories."""
        model = flow_model if flow_model is not None else self.model
        B = x_t.shape[0]
        # diffusion_step_encoder accepts (B, 1, 1) or (B,) style inputs like CFM path
        if t_scalar.ndim == 0:
            t_in = t_scalar.reshape(1).expand(B)
            timeembedding = model.diffusion_step_encoder(t_in)
        elif t_scalar.ndim == 1:
            timeembedding = model.diffusion_step_encoder(t_scalar)
        else:
            timeembedding = model.diffusion_step_encoder(t_scalar)
            timeembedding = timeembedding.reshape(B, -1)
        if timeembedding.ndim == 1:
            timeembedding = timeembedding.unsqueeze(0).expand(B, -1)
        elif timeembedding.shape[0] == 1 and B > 1:
            timeembedding = timeembedding.expand(B, -1)

        network_output = model(x_t, timeembedding, obs_cond)
        network_output = self.config.mlp_output_scale * network_output
        if self.config.transported_clip_value is not None:
            network_output = network_output.clamp(
                -self.config.transported_clip_value, self.config.transported_clip_value
            )

        if self.config.flow_network_output_param == "u":
            return network_output
        # x0 parameterization: u = (x_t - x0) / t
        x0_pred = network_output
        t_for_div = t_scalar
        while t_for_div.ndim < x_t.ndim:
            t_for_div = t_for_div.unsqueeze(-1)
        t_clamped = torch.clamp(t_for_div, min=1e-5)
        return (x_t - x0_pred) / t_clamped

    def _integrate_flow(
        self,
        obs_cond: Tensor,
        x_t: Tensor,
        n_steps: int | None = None,
        integration_model: nn.Module | None = None,
    ) -> Tensor:
        """Euler-integrate flow from noise x_t (t=1) to data (t=0) in normalized action space.

        Optionally integrate with a detached copy (e.g. EMA snapshot). Never mutate the
        online ``self.model`` here — that path destabilized ``fpo_operator`` training.
        """
        flow_model = integration_model if integration_model is not None else self.model
        flow_steps = int(n_steps) if n_steps is not None else int(self.config.sampling_steps)
        flow_steps = max(flow_steps, 1)
        t_path = torch.linspace(1.0, 0.0, flow_steps + 1, device=x_t.device)
        x = x_t
        for i in range(flow_steps):
            t_current = t_path[i]
            t_next = t_path[i + 1]
            dt = t_next - t_current
            velocity = self._predict_velocity(
                x, t_current, obs_cond, flow_model=flow_model
            )
            x = x + velocity * dt
        return x

    def get_reflow_loss(
        self,
        batch: dict[str, Tensor],
        n_samples_per_obs: int = 4,
        advantages: Tensor | None = None,
        reflow_mode: str = "uniform",
        advantage_threshold: float = 0.0,
        use_ema_endpoint: bool = False,
        ema_warmup_updates: int = 500,
        zero_x0_prob: float = 0.0,
    ) -> Tensor:
        """Rectified-flow reflow aux loss on straight paths (noise -> flow endpoint).

        Operates in normalized action space, matching ``get_cfm_loss``.
        Endpoint integration may use a frozen teacher or detached EMA copy;
        velocity regression always uses the online model.
        """
        batch = self.normalize_inputs(batch)
        obs_cond = self.model.encode_observations(batch)
        B = obs_cond.shape[0]
        H = self.config.horizon
        D = self.model.action_dim
        device = obs_cond.device

        x0 = torch.randn(B, n_samples_per_obs, H, D, device=device)
        if zero_x0_prob > 0.0:
            zero_mask = torch.rand(B, n_samples_per_obs, 1, 1, device=device) < zero_x0_prob
            x0 = torch.where(zero_mask, torch.zeros_like(x0), x0)

        integration_model = self._reflow_integration_model(
            use_ema_endpoint, ema_warmup_updates
        )
        endpoint_obs = obs_cond
        teacher = self.__dict__.get("_teacher_model")
        if teacher is not None:
            integration_model = teacher
            with torch.no_grad():
                endpoint_obs = teacher.encode_observations(batch)

        endpoint_obs_flat = (
            endpoint_obs[:, None, :]
            .expand(B, n_samples_per_obs, -1)
            .reshape(B * n_samples_per_obs, -1)
        )
        x0_flat = x0.reshape(B * n_samples_per_obs, H, D)

        with torch.no_grad():
            x1_prime = self._integrate_flow(
                endpoint_obs_flat,
                x0_flat,
                n_steps=self.config.sampling_steps,
                integration_model=integration_model,
            )
        x1_prime = x1_prime.reshape(B, n_samples_per_obs, H, D)

        t = torch.rand(B, n_samples_per_obs, 1, 1, device=device)
        x_t = t * x0 + (1.0 - t) * x1_prime
        target_velocity = x0 - x1_prime

        obs_cond_exp = (
            obs_cond[:, None, :]
            .expand(B, n_samples_per_obs, -1)
            .reshape(B * n_samples_per_obs, -1)
        )
        velocity_pred = self._predict_velocity(
            x_t.reshape(B * n_samples_per_obs, H, D),
            t.reshape(B * n_samples_per_obs, 1, 1),
            obs_cond_exp,
        ).reshape(B, n_samples_per_obs, H, D)

        per_sample_loss = self._compute_squared_error(velocity_pred, target_velocity)
        # mean over horizon/action dims -> (B, n_samples)
        per_sample_loss = per_sample_loss.mean(dim=(-1, -2))

        if reflow_mode in {"reward_aware", "fpo_operator"} and advantages is not None:
            sample_weights = compute_advantage_reflow_weights(
                advantages, reflow_mode, advantage_threshold
            )
            obs_weights = sample_weights[:, None].expand_as(per_sample_loss)
            return (obs_weights * per_sample_loss).sum() / (obs_weights.sum() + 1e-8)
        return per_sample_loss.mean()

    def get_teacher_kd_loss(
        self,
        batch: dict[str, Tensor],
        step_bins: list[int] | None = None,
        zero_x0_prob: float = 0.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """Match student few-step actions to a frozen teacher's full-step action.

        Same x0 for teacher and student. Teacher endpoints are stopgrad.
        """
        teacher = self.__dict__.get("_teacher_model")
        device = next(self.model.parameters()).device
        if teacher is None:
            zero = torch.tensor(0.0, device=device)
            return zero, {}

        max_steps = int(self.config.sampling_steps)
        bins = list(step_bins) if step_bins is not None else [1, 4, 8]
        bins = [int(s) for s in bins if 0 < int(s) < max_steps]
        if not bins:
            zero = torch.tensor(0.0, device=device)
            return zero, {}

        batch = self.normalize_inputs(batch)
        obs_student = self.model.encode_observations(batch)
        with torch.no_grad():
            obs_teacher = teacher.encode_observations(batch)

        B = obs_student.shape[0]
        H = self.config.horizon
        D = self.model.action_dim
        x0 = torch.randn(B, H, D, device=obs_student.device)
        if zero_x0_prob > 0.0:
            zero_mask = torch.rand(B, 1, 1, device=obs_student.device) < zero_x0_prob
            x0 = torch.where(zero_mask, torch.zeros_like(x0), x0)

        with torch.no_grad():
            action_teacher = self._integrate_flow(
                obs_teacher,
                x0,
                n_steps=max_steps,
                integration_model=teacher,
            )

        total_loss = torch.tensor(0.0, device=obs_student.device)
        metrics: dict[str, float] = {}
        for steps in bins:
            action_low = self._integrate_flow(obs_student, x0, n_steps=int(steps))
            step_loss = (action_low - action_teacher).pow(2).mean()
            total_loss = total_loss + step_loss
            metrics[f"teacher_kd_{steps}"] = float(step_loss.item())

        loss = total_loss / float(len(bins))
        metrics["teacher_kd_loss"] = float(loss.item())
        return loss, metrics

    def get_adaptive_compute_loss(
        self,
        batch: dict[str, Tensor],
        advantages: Tensor | None = None,
        latency_penalty_coef: float = 1.0,
        step_bins: list[int] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Train step predictor: match low-step actions to full-step with latency penalty."""
        bins = list(step_bins) if step_bins is not None else [1, 2, 4, 8]
        max_steps = int(self.config.sampling_steps)
        bins = [int(s) for s in bins if 0 < int(s) < max_steps]
        if not bins:
            zero = torch.tensor(0.0, device=next(self.model.parameters()).device)
            return zero, {}

        predictor = self.ensure_step_predictor(bins)
        batch = self.normalize_inputs(batch)
        obs_cond = self.model.encode_observations(batch)
        B = obs_cond.shape[0]
        H = self.config.horizon
        D = self.model.action_dim
        device = obs_cond.device

        x0 = torch.zeros(B, H, D, device=device)
        with torch.no_grad():
            action_full = self._integrate_flow(obs_cond, x0, n_steps=max_steps)

        pred_steps = predictor.predict_steps(obs_cond)
        action_low = torch.zeros_like(action_full)
        for steps in bins:
            mask = pred_steps == steps
            if not mask.any():
                continue
            action_low[mask] = self._integrate_flow(
                obs_cond[mask], x0[mask], n_steps=int(steps)
            )

        fidelity_loss = (action_low - action_full).pow(2).mean()
        latency_penalty = predictor.expected_step_fraction(obs_cond).mean()
        adv_bonus = torch.tensor(0.0, device=device)
        if advantages is not None:
            adv_bonus = torch.clamp(advantages.reshape(-1), min=0.0).mean()

        loss = fidelity_loss + latency_penalty_coef * latency_penalty - 0.1 * adv_bonus
        metrics = {
            "adaptive_fidelity_loss": float(fidelity_loss.item()),
            "adaptive_latency_penalty": float(latency_penalty.item()),
            "adaptive_mean_steps": float(pred_steps.float().mean().item()),
        }
        return loss, metrics

    @torch.no_grad()
    def compute_theory_metrics(
        self,
        batch: dict[str, Tensor],
        n_samples_per_obs: int = 1,
    ) -> dict[str, float]:
        """Theory hooks: path straightness and discretization gap proxies."""
        batch = self.normalize_inputs(batch)
        obs_cond = self.model.encode_observations(batch)
        B = obs_cond.shape[0]
        H = self.config.horizon
        D = self.model.action_dim
        device = obs_cond.device
        max_steps = int(self.config.sampling_steps)

        x0 = torch.zeros(B, n_samples_per_obs, H, D, device=device)
        obs_flat = (
            obs_cond[:, None, :]
            .expand(B, n_samples_per_obs, -1)
            .reshape(B * n_samples_per_obs, -1)
        )
        x0_flat = x0.reshape(B * n_samples_per_obs, H, D)
        x1 = self._integrate_flow(obs_flat, x0_flat, n_steps=max_steps)
        x1 = x1.reshape(B, n_samples_per_obs, H, D)

        t_mid = torch.full((B, n_samples_per_obs, 1, 1), 0.5, device=device)
        x_t = t_mid * x0 + (1.0 - t_mid) * x1
        target_velocity = x0 - x1
        velocity_pred = self._predict_velocity(
            x_t.reshape(B * n_samples_per_obs, H, D),
            t_mid.reshape(B * n_samples_per_obs, 1, 1),
            obs_flat,
        ).reshape(B, n_samples_per_obs, H, D)

        straightness = compute_path_straightness(x0, x1, velocity_pred).mean().item()
        metrics = {"path_straightness": float(straightness)}
        for low_steps in (1, 2, 4, 8):
            if low_steps >= max_steps:
                continue
            x1_low = self._integrate_flow(obs_flat, x0_flat, n_steps=low_steps)
            x1_low = x1_low.reshape(B, n_samples_per_obs, H, D)
            gap = compute_discretization_gap(x1, x1_low).mean().item()
            metrics[f"discretization_gap_{low_steps}"] = float(gap)
        return metrics


