#!/usr/bin/env python

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
from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode


@PreTrainedConfig.register_subclass("flowmatching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    """Configuration class for the Flow Matching Policy.

    Flow Matching Policy uses continuous normalizing flows to predict actions.
    It learns a vector field that transports noise to the data distribution.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        horizon: The prediction horizon (chunk size) in units of environment steps.
        n_action_steps: The number of action steps to execute in the environment.
        sampling_steps: Number of integration steps for inference (Euler integration).
        timestep_embed_dim: Dimension of the timestep embedding.
        down_dims: Dimensions for the U-Net encoder layers.
        kernel_size: Kernel size for convolutional layers.
        n_groups: Number of groups for group normalization.
        cond_predict_scale: Whether to use conditional prediction scaling.
        flow_network_output_param: Output parameterization ('x0' or 'u' for velocity).
        cfm_loss_mode: Loss mode ('x0', 'u' for velocity, or 'eps').
        cfm_loss_weight_from_t: Weighting scheme for loss over timesteps.
        mlp_output_scale: Scale factor for network output.
        actor_scale: Scale factor for final actions.
        exploration_noise_std: Standard deviation for exploration noise (training only).
        input_shapes: Dictionary defining the shapes of the input data.
        output_shapes: Dictionary defining the shapes of the output data.
        input_normalization_modes: Normalization modes for inputs.
        output_normalization_modes: Normalization modes for outputs.
        vision_backbone: Name of the vision backbone to use.
        pretrained_backbone_weights: Pretrained weights for the vision backbone.
        obs_as_global_cond: Whether to use observations as global conditioning.
        crop_shape: Shape to crop images to (if specified).
        obs_encoder_group_norm: Whether to use group norm in observation encoder.
        eval_fixed_crop: Whether to use fixed crop during evaluation.
        ema_power: Exponential moving average power for model weights.
        dropout: Dropout rate.
        action_dropout: Action-specific dropout rate.
    """

    # Input / output structure
    # n_obs_steps: int = 2  # Number of observation steps # not used yet
    horizon: int = 16  # Prediction horizon (action chunk size)
    n_action_steps: int = 8  # Number of action steps to execute

    # Flow matching parameters
    sampling_steps: int = 10  # Number of Euler integration steps for inference
    timestep_embed_dim: int = 32  # Dimension of timestep embedding

    # Flow network configuration
    flow_network_output_param: str = "u" #Literal["x0", "u"] = "u"  # Output parameterization
    cfm_loss_mode: str = "u" #Literal["x0", "u", "eps"] = "u"  # Loss mode
    cfm_loss_weight_from_t: str = "constant" #Literal["constant", "linear_1_to_0.1", "linear_1_to_0.01"] = "constant"
    cfm_loss_use_huber: bool = True
    cfm_loss_huber_delta: float = 0.5
    mlp_output_scale: float = 1.0  # Scale network output
    actor_scale: float = 1.0  # Scale final actions
    exploration_noise_std: float = 0.0  # Exploration noise (training only)
    transported_clip_value: float | None = None  # Clip transported predictions (x0 or u) to [-value, value]. None means no clipping

    # Architecture selection
    network_architecture: str = "unet"  # "unet" or "mlp"
    """Network architecture type: 'unet' for 1D-CNN based, 'mlp' for MLP based."""

    # U-Net architecture parameters (used when network_architecture="unet")
    down_dims: list[int] = field(default_factory=lambda: [256, 512, 1024])
    kernel_size: int = 5
    n_groups: int = 8  # Number of groups for GroupNorm
    cond_predict_scale: bool = True

    # MLP architecture parameters (used when network_architecture="mlp")
    mlp_dims: list[int] = field(default_factory=lambda: [512, 512, 512])
    """Hidden dimensions for MLP layers when using MLP architecture."""

    # Vision encoder parameters
    vision_backbone: str = "resnet18"  # Options: "resnet18", "resnet34", "resnet50", "vit"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    obs_encoder_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32

    # ViT-specific parameters (used when vision_backbone="vit")
    image_size: int = 84  # Image size (assumes square images)
    vit_patch_size: int = 8  # Patch size for ViT
    vit_depth: int = 1  # Number of transformer layers
    vit_embed_dim: int = 128  # Embedding dimension
    vit_num_heads: int = 4  # Number of attention heads

    # Image processing
    crop_shape: list[int] | None = field(default_factory=lambda: [84, 84])
    eval_fixed_crop: bool = True

    # Observation conditioning
    obs_as_global_cond: bool = True

    # Normalization - use FeatureType keys
    normalization_mapping: dict[FeatureType, NormalizationMode] = field(
        default_factory=lambda: {
            FeatureType.VISUAL: NormalizationMode.MEAN_STD,
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        }
    )

    # Training parameters
    ema_power: float = 0.75  # Exponential moving average for model weights
    dropout: float = 0.0
    action_dropout: float = 0.0

    # Optimizer parameters
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-6
    optimizer_lr_backbone: float = 1e-5

    # Input/output configuration
    input_features: list = field(default_factory=list)
    output_features: list = field(default_factory=lambda: ["action"])
    input_shapes: dict = field(default_factory=dict)
    output_shapes: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        # Validation
        if self.n_action_steps > self.horizon:
            raise ValueError(f"n_action_steps ({self.n_action_steps}) cannot be greater than horizon ({self.horizon})")

        valid_backbones = ["resnet18", "resnet34", "resnet50", "vit"]
        if not (self.vision_backbone.startswith("resnet") or self.vision_backbone == "vit"):
            raise ValueError(f"vision_backbone must be one of {valid_backbones}. Got {self.vision_backbone}")

        if self.flow_network_output_param not in ["x0", "u"]:
            raise ValueError(f"flow_network_output_param must be 'x0' or 'u'. Got {self.flow_network_output_param}")

        if self.cfm_loss_mode not in ["x0", "u", "eps"]:
            raise ValueError(f"cfm_loss_mode must be 'x0', 'u', or 'eps'. Got {self.cfm_loss_mode}")

        if self.network_architecture not in ["unet", "mlp", "residual_mlp"]:
            raise ValueError(f"network_architecture must be 'unet', 'mlp', or 'residual_mlp'. Got {self.network_architecture}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        # Could return a cosine annealing scheduler here if desired
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.state_features:
            raise ValueError("You must provide at least one image or state observation among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def image_features(self) -> list:
        """Return list of image feature keys."""
        if not hasattr(self, "_image_features"):
            self._image_features = []
        return self._image_features

    @image_features.setter
    def image_features(self, value: list):
        """Set image feature keys."""
        self._image_features = value

    @property
    def state_features(self) -> list:
        """Return list of state feature keys."""
        if not hasattr(self, "_state_features"):
            self._state_features = []
        return self._state_features

    @state_features.setter
    def state_features(self, value: list):
        """Set state feature keys."""
        self._state_features = value
