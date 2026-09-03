#!/usr/bin/env python

"""Flow Matching Policy with MLP Architecture"""

import math
from typing import Union

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

from .flow_model_config import FlowMatchingConfig
from .vit import VitEncoder, VitEncoderConfig



class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


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

class ResidualBlock(nn.Module):
    """Residual block for MLP."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.Mish()

    def forward(self, x):
        return self.activation(self.linear(x) + x)

class FlowMatchingResidualMLPModel(nn.Module):
    """MLP-based Flow Matching model for behavior cloning."""

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        # Get action dimension from output shapes
        self.action_dim = 2  # Default, will be overridden from config if available
        if hasattr(config, "output_shapes") and "action" in config.output_shapes:
            shape = config.output_shapes["action"]
            if isinstance(shape, (list, tuple)):
                self.action_dim = shape[-1] if len(shape) > 0 else 2
            else:
                self.action_dim = shape

        # Vision encoder
        self.vision_encoder = self._init_vision_encoder(config)
        self.vision_feature_dim = self._get_vision_feature_dim(config.vision_backbone)

        # Calculate observation dimension
        obs_dim = 0
        if config.image_features:
            obs_dim += self.vision_feature_dim * len(config.image_features)
        if config.state_features:
            for f in config.state_features:
                if f in config.input_shapes:
                    shape = config.input_shapes[f]
                    if isinstance(shape, (list, tuple)):
                        obs_dim += shape[-1] if len(shape) > 0 else 1
                    else:
                        obs_dim += shape

        # Global conditioning dimension
        self.global_cond_dim = obs_dim

        # Time embedding dimension
        time_embed_dim = config.timestep_embed_dim

        # MLP architecture
        # Input: [noisy_action (action_dim * horizon), time_embedding, observation_conditioning]
        input_dim = self.action_dim * config.horizon + time_embed_dim + self.global_cond_dim

        # Get MLP dimensions from config, default to [512, 512, 512]
        mlp_dims = getattr(config, 'mlp_dims', [512, 512, 512])

        # Build MLP layers
        layers = [nn.Linear(input_dim, mlp_dims[0])]
        prev_dim = mlp_dims[0]

        for hidden_dim in mlp_dims[1:]:
            # layers.append(nn.Linear(prev_dim, hidden_dim))
            # layers.append(nn.Mish())
            # Add residual connection layer between previous and current layer
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.action_dim * config.horizon))

        self.mlp = nn.Sequential(*layers)

        # Time step encoder (sinusoidal positional encoding)
        self.diffusion_step_encoder= SinusoidalPosEmb(time_embed_dim)


    def _init_vision_encoder(self, config):
        """Initialize the vision encoder."""
        if config.vision_backbone.startswith("resnet"):
            encoder = get_resnet(config.vision_backbone, weights=config.pretrained_backbone_weights)
            if config.obs_encoder_group_norm:
                encoder = replace_bn_with_gn(encoder)
            return encoder
        elif config.vision_backbone == "vit":
            # Infer image size from actual input shapes in the dataset
            img_h, img_w = None, None
            if config.image_features and len(config.image_features) > 0:
                first_img_key = config.image_features[0]
                if first_img_key in config.input_shapes:
                    img_shape = config.input_shapes[first_img_key]
                    # Shape could be (H, W, C) or (C, H, W)
                    if isinstance(img_shape, (list, tuple)) and len(img_shape) == 3:
                        if img_shape[0] == 3 or img_shape[0] == 1:
                            # (C, H, W) format
                            img_h, img_w = img_shape[1], img_shape[2]
                        else:
                            # (H, W, C) format
                            img_h, img_w = img_shape[0], img_shape[1]

            # Fallback to config or default
            if img_h is None or img_w is None:
                img_size = getattr(config, "image_size", 84)
                img_h, img_w = img_size, img_size
                print(f"[ViT Init] Using default/config image size: {img_h}x{img_w}")
            else:
                print(f"[ViT Init] Inferred image size from dataset: {img_h}x{img_w}")

            vit_config = VitEncoderConfig(
                patch_size=getattr(config, "vit_patch_size", 8),
                depth=getattr(config, "vit_depth", 1),
                embed_dim=getattr(config, "vit_embed_dim", 128),
                num_heads=getattr(config, "vit_num_heads", 4),
            )
            encoder = VitEncoder(
                obs_shape=[3, img_h, img_w],
                cfg=vit_config,
                num_channel=3,
                img_h=img_h,
                img_w=img_w,
            )
            return encoder
        raise ValueError(f"Unsupported vision backbone: {config.vision_backbone}")

    def _get_vision_feature_dim(self, backbone_name):
        """Get the output dimension of the vision encoder."""
        if backbone_name == "resnet18" or backbone_name == "resnet34":
            return 512
        if backbone_name == "resnet50":
            return 2048
        if backbone_name == "vit":
            # For ViT, we flatten the patches: repr_dim = embed_dim * num_patches
            return self.vision_encoder.repr_dim
        return 512  # Default

    def encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode observations into a conditioning vector."""
        B = next(iter(batch.values())).shape[0]
        features = []

        # Encode image observations
        if self.config.image_features:
            for img_key in self.config.image_features:
                if img_key in batch:
                    img_tensor = batch[img_key]
                    # img_tensor shape: (B, T, C, H, W) or (B, C, H, W)
                    if len(img_tensor.shape) == 5:
                        B, T = img_tensor.shape[:2]
                        img_flat = img_tensor.flatten(end_dim=1)  # (B*T, C, H, W)
                    else:
                        B = img_tensor.shape[0]
                        T = 1
                        img_flat = img_tensor  # Already (B, C, H, W)

                    # Pass through vision encoder
                    if self.config.vision_backbone == "vit":
                        # ViT expects 0-255 input and returns (B, num_patches, embed_dim)
                        # We need to flatten to (B, repr_dim)
                        img_features = self.vision_encoder(img_flat * 255.0, flatten=True)  # (B*T, repr_dim) or (B, repr_dim)
                    else:
                        # ResNet already outputs (B, D)
                        img_features = self.vision_encoder(img_flat)  # (B*T, D) or (B, D)

                    # Reshape back
                    if len(img_tensor.shape) == 5:
                        img_features = img_features.reshape(B, T, -1)  # (B, T, D)
                    else:
                        img_features = img_features.reshape(B, 1, -1)  # (B, 1, D)
                    features.append(img_features)

        # Encode state observations
        if self.config.state_features:
            for state_key in self.config.state_features:
                if state_key in batch:
                    state_tensor = batch[state_key]  # (B, T, D) or (B, D)
                    # Ensure 3D tensor (B, T, D)
                    if len(state_tensor.shape) == 2:
                        state_tensor = state_tensor.unsqueeze(1)  # (B, 1, D)
                    features.append(state_tensor)

        # Concatenate all features
        if features:
            # Ensure all features have the same temporal dimension
            max_T = max(f.shape[1] for f in features)
            padded_features = []
            for f in features:
                if f.shape[1] < max_T:
                    # Repeat last frame to match temporal dimension
                    padding = f[:, -1:, :].repeat(1, max_T - f.shape[1], 1)
                    f = torch.cat([f, padding], dim=1)
                padded_features.append(f)

            obs_features = torch.cat(padded_features, dim=-1)  # (B, T, total_dim)
            # Flatten temporal dimension for global conditioning
            obs_cond = obs_features.flatten(start_dim=1)  # (B, T * total_dim)
        else:
            obs_cond = torch.zeros((B, 1), device=next(iter(batch.values())).device)

        return obs_cond

    def forward(self, x_t: Tensor, t_emb: Tensor, obs_cond: Tensor) -> Tensor:
        """
        Forward pass through the MLP.

        Args:
            x_t: (B, T, D) noisy actions where T is horizon and D is action_dim
            t_emb: (B, time_embed_dim) time embedding (already encoded)
            obs_cond: (B, obs_dim) observation conditioning

        Returns:
            (B, T, D) predicted velocity or x0
        """
        B, T, D = x_t.shape

        # Flatten actions to (B, T*D)
        x_t_flat = x_t.reshape(B, -1)

        # Concatenate all inputs (t_emb is already encoded)
        mlp_input = torch.cat([x_t_flat, t_emb, obs_cond], dim=-1)  # (B, input_dim)

        # Pass through MLP
        output = self.mlp(mlp_input)  # (B, T*D)

        # Reshape back to (B, T, D)
        output = output.reshape(B, T, D)

        return output

    def initialize_layers(self, init_fn=None):
        """
        Initialize all layers in the MLP with a given initialization function.
        If no function is provided, uses Kaiming normal initialization for Linear layers.
        """
        def default_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        init = init_fn if init_fn is not None else default_init

        # Initialize MLP layers
        self.mlp.apply(init)
        # Initialize time encoder
        self.diffusion_step_encoder.apply(init)

