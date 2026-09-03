# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flow Matching Policy with UNet Architecture"""

import math
from typing import Union

import torch
import torchvision
from torch import Tensor, nn

from .flow_model_config import FlowMatchingConfig
from .vit import VitEncoder, VitEncoderConfig

# Helper modules (reuse from mydiffusion implementation)

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


class Downsample1d(nn.Module):
    """1D downsampling layer."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """1D upsampling layer."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish activation block."""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Conditional residual block with FiLM modulation."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1)))

        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)

        # Apply FiLM conditioning
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """1D U-Net with conditioning for diffusion models."""

    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    ):
        super().__init__()

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Diffusion timestep embedding
        dsed = diffusion_step_embed_dim
        cond_dim = dsed + global_cond_dim

        # Encoder (downsampling)
        self.down_modules = nn.ModuleList([])
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Middle layers
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
            ]
        )

        # Decoder (upsampling)
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Final convolution
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def initialize_layers(self, init_fn=None):
        """
        Initialize all layers in the U-Net with a given initialization function.
        If no function is provided, uses Kaiming normal initialization for Conv1d/Linear layers,
        and handles GroupNorm and BatchNorm layers.
        """
        def default_init(m):
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        init = init_fn if init_fn is not None else default_init

        # Initialize down modules
        for module_list in self.down_modules:
            for m in module_list:
                m.apply(init)
        # Initialize mid modules
        for m in self.mid_modules:
            m.apply(init)
        # Initialize up modules if present
        if hasattr(self, "up_modules"):
            for module_list in self.up_modules:
                for m in module_list:
                    m.apply(init)
        # Initialize diffusion step encoder
        self.diffusion_step_encoder.apply(init)

    # def forward(self, sample: Tensor, timestep: Union[Tensor, float], global_cond=None):
    def forward(self, sample: Tensor, timeembedding: Union[Tensor, float], global_cond=None):
        """
        Forward pass through the U-Net.

        Args:
            sample: (B, T, C) tensor of noisy actions
            # timestep: (B,) tensor or float of diffusion timesteps
            global_cond: (B, D) tensor of global conditioning

        Returns:
            (B, T, C) tensor of predicted noise
        """
        # Reshape to (B, C, T) for Conv1d
        sample = sample.moveaxis(-1, -2)

        if global_cond is not None:
            global_feature = torch.cat([timeembedding, global_cond], axis=-1)
        else:
            global_feature = timeembedding

        # Encoder
        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Middle
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Decoder
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        # Final conv
        x = self.final_conv(x)

        # Reshape back to (B, T, C)
        x = x.moveaxis(-1, -2)
        return x


class FlowMatchingUnetModel(nn.Module):
    """UNet-based Flow Matching model for behavior cloning."""

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
        global_cond_dim = obs_dim

        # UNet architecture
        self.unet = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=config.timestep_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
        )

        # For compatibility with the rest of the code
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.timestep_embed_dim * 4, config.timestep_embed_dim),
        )

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
        Forward pass through the UNet.

        Args:
            x_t: (B, T, D) noisy actions where T is horizon and D is action_dim
            t_emb: (B, time_embed_dim) time embedding (already encoded)
            obs_cond: (B, obs_dim) observation conditioning

        Returns:
            (B, T, D) predicted velocity or x0
        """
        # Pass through UNet (it expects timeembedding and global_cond separately)
        output = self.unet(x_t, t_emb, obs_cond)

        return output

    def initialize_layers(self, init_fn=None):
        """
        Initialize all layers in the UNet with a given initialization function.
        Delegates to the UNet's initialize_layers method.
        """
        self.unet.initialize_layers(init_fn)


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

