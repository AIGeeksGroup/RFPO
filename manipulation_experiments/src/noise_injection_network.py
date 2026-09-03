#!/usr/bin/env python

"""Noise Injection Network for learned exploration noise in Flow Matching Policy."""

import math
import torch
import torch.nn as nn
from torch import Tensor


class NoiseInjectionNetwork(nn.Module):
    """
    Network that predicts state-dependent noise injection scale for flow matching.

    Takes observation conditioning as input and outputs noise scale tensor
    with shape (B, T, 1, D) where:
        - B: batch size
        - T: action horizon (chunk size)
        - 1: singleton dimension for broadcasting across flow sampling steps
        - D: action dimension

    The output represents the standard deviation of exploration noise to inject,
    conditioned on the current observation. The singleton dimension allows
    broadcasting across the flow sampling steps dimension.
    """

    def __init__(
        self,
        obs_cond_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: list[int] = None,
        min_noise: float = 1e-4,
        max_noise: float = 1.0,
        device: str = "cuda",
    ):
        """
        Initialize the noise injection network.

        Args:
            obs_cond_dim: Dimension of the observation conditioning vector
            action_dim: Dimension of the action space (D)
            horizon: Action horizon / chunk size (T)
            hidden_dims: Hidden layer dimensions for the MLP
            min_noise: Minimum noise scale (for numerical stability)
            max_noise: Maximum noise scale (for bounded exploration)
        """
        super().__init__()

        self.obs_cond_dim = obs_cond_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.device = device
        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Input: obs_cond
        input_dim = obs_cond_dim

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim

        # Output layer: predict noise scale for each (horizon, action_dim) pair
        # Output shape after reshape: (B, T, D)
        layers.append(nn.Linear(prev_dim, horizon * action_dim))

        self.mlp = nn.Sequential(*layers)

        self.logvar_min = torch.nn.Parameter(torch.log(torch.tensor(min_noise**2, dtype=torch.float32, device=self.device)), requires_grad=False)
        self.logvar_max = torch.nn.Parameter(torch.log(torch.tensor(max_noise**2, dtype=torch.float32, device=self.device)), requires_grad=False)
    
        # Initialize output layer to produce small initial noise
        self._init_output_layer()

    def _init_output_layer(self):
        """Initialize the output layer to produce conservative initial noise."""
        # Get the last linear layer
        for module in reversed(list(self.mlp.modules())):
            if isinstance(module, nn.Linear):
                # Initialize to produce noise close to min_noise after sigmoid scaling
                # sigmoid(bias) * (max - min) + min ≈ min_noise
                # We want sigmoid(bias) ≈ 0, so bias should be negative
                nn.init.zeros_(module.weight)
                # Start with small negative bias to produce low noise initially
                nn.init.constant_(module.bias, -3.0)
                break

    def forward(self, obs_cond: Tensor) -> Tensor:
        """
        Forward pass to predict noise injection scales.

        Args:
            obs_cond: Observation conditioning tensor of shape (B, obs_cond_dim)

        Returns:
            Noise scale tensor of shape (B, T, 1, D) where T is horizon
            and D is action_dim. The singleton dimension allows broadcasting
            across flow sampling steps.
        """
        B = obs_cond.shape[0]

        # Pass through MLP: (B, obs_cond_dim) -> (B, T * D)
        noise_logits = self.mlp(obs_cond)

        # Reshape: (B, T * D) -> (B, T, D)
        noise_logits = noise_logits.reshape(B, self.horizon, self.action_dim)

        noise_logvar = torch.tanh(noise_logits)
        noise_logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * (noise_logvar + 1)/2.0
        noise_std = torch.exp(0.5 * noise_logvar)

        # Add singleton dimension: (B, T, D) -> (B, T, 1, D)
        # This allows broadcasting across flow sampling steps
        noise_std = noise_std.unsqueeze(2)
        return noise_std


