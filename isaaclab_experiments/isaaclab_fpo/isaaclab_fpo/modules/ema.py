"""
Exponential Moving Average (EMA) for model weights.

EMA maintains a smoothed copy of model parameters that is updated at each training step:
    ema_param = decay * ema_param + (1 - decay) * current_param

This is particularly useful for generative models (like flow matching) where the EMA model
often produces better samples than the currently-being-trained model.
"""

import torch
import torch.nn as nn
from typing import Optional


class ExponentialMovingAverage:
    """
    Maintains exponential moving average of model parameters.

    Args:
        model: The model whose parameters to track
        decay: The EMA decay rate (e.g., 0.95, 0.99, 0.999)
        device: Device to store EMA parameters on
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.95,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device

        # Create a copy of the model parameters for EMA
        self.shadow_params = {}
        self.model_params = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Store reference to original parameter
                self.model_params[name] = param
                # Create EMA shadow copy
                self.shadow_params[name] = param.data.clone().to(self.device)

    @torch.no_grad()
    def update(self):
        """
        Update the EMA parameters based on current model parameters.
        Should be called after each optimizer step.
        """
        for name, param in self.model_params.items():
            if param.requires_grad:
                # EMA update: ema = decay * ema + (1 - decay) * current
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data.to(self.device), alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def reset_to_current(self):
        """
        Reset EMA shadow parameters to current model parameters.
        Called at warmup step to initialize EMA with trained weights instead of initial weights.
        """
        for name, param in self.model_params.items():
            if param.requires_grad:
                self.shadow_params[name].copy_(param.data.to(self.device))

    def state_dict(self):
        """
        Returns the EMA state dict for checkpointing.
        """
        return {
            'decay': self.decay,
            'shadow_params': self.shadow_params,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the EMA state from a checkpoint.
        """
        self.decay = state_dict['decay']
        self.shadow_params = state_dict['shadow_params']

    def copy_to(self, model: nn.Module):
        """
        Copy EMA parameters to the model.
        Useful for evaluation or saving checkpoints with EMA weights.

        Args:
            model: The model to copy EMA parameters to
        """
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    def store(self, model: nn.Module):
        """
        Save current model parameters (before copying EMA parameters).
        Use with restore() to temporarily use EMA weights for evaluation.

        Args:
            model: The model whose parameters to store
        """
        self.backup_params = {}
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                self.backup_params[name] = param.data.clone()

    def restore(self, model: nn.Module):
        """
        Restore model parameters (after using EMA weights).
        Use with store() to temporarily use EMA weights for evaluation.

        Args:
            model: The model whose parameters to restore
        """
        for name, param in model.named_parameters():
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params = {}

    def get_ema_model_state_dict(self):
        """
        Get a state dict with EMA parameters suitable for saving.
        Returns a dict in the same format as model.state_dict().
        """
        return {name: param.clone() for name, param in self.shadow_params.items()}
