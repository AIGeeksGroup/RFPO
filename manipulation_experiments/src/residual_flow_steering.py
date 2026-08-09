from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass(frozen=True)
class RFSAction:
    latent: Tensor
    residual: Tensor
    latent_raw: Tensor
    residual_raw: Tensor
    log_prob: Tensor
    log_prob_steps: Tensor
    entropy: Tensor


class ResidualFlowSteeringPolicy(nn.Module):
    """Chunk-level PPO policy for joint flow-source and residual modulation."""

    def __init__(
        self,
        observation_dim: int,
        horizon: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        latent_mean_limit: float = 2.0,
        residual_scale: float = 0.1,
        latent_log_std_init: float = 0.0,
        residual_log_std_init: float = -1.5,
    ) -> None:
        super().__init__()
        if min(observation_dim, horizon, action_dim) <= 0:
            raise ValueError("RFS dimensions must be positive")
        if not hidden_dims or min(hidden_dims) <= 0:
            raise ValueError("RFS hidden dimensions must be positive")
        if latent_mean_limit <= 0 or residual_scale <= 0:
            raise ValueError("RFS output scales must be positive")

        self.horizon = horizon
        self.action_dim = action_dim
        self.chunk_dim = horizon * action_dim
        self.latent_mean_limit = latent_mean_limit
        self.residual_scale = residual_scale

        layers: list[nn.Module] = []
        input_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, 2 * self.chunk_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

        self.latent_log_std = nn.Parameter(
            torch.full((self.chunk_dim,), latent_log_std_init)
        )
        self.residual_log_std = nn.Parameter(
            torch.full((self.chunk_dim,), residual_log_std_init)
        )

    def _distributions(self, observations: Tensor) -> tuple[Normal, Normal]:
        features = self.trunk(observations)
        latent_raw_mean, residual_raw_mean = self.mean_head(features).chunk(2, dim=-1)
        latent_mean = self.latent_mean_limit * torch.tanh(latent_raw_mean)
        latent_std = self.latent_log_std.clamp(-5.0, 2.0).exp()
        residual_std = self.residual_log_std.clamp(-5.0, 1.0).exp()
        return (
            Normal(latent_mean, latent_std),
            Normal(residual_raw_mean, residual_std),
        )

    def sample(self, observations: Tensor, deterministic: bool = False) -> RFSAction:
        latent_dist, residual_dist = self._distributions(observations)
        if deterministic:
            latent_raw = latent_dist.mean
            residual_raw = residual_dist.mean
        else:
            latent_raw = latent_dist.sample()
            residual_raw = residual_dist.sample()
        return self.evaluate_actions(observations, latent_raw, residual_raw)

    def evaluate_actions(
        self,
        observations: Tensor,
        latent_raw: Tensor,
        residual_raw: Tensor,
    ) -> RFSAction:
        expected_shape = (observations.shape[0], self.chunk_dim)
        if latent_raw.shape != expected_shape or residual_raw.shape != expected_shape:
            raise ValueError(
                f"RFS raw actions must both have shape {expected_shape}; got "
                f"{tuple(latent_raw.shape)} and {tuple(residual_raw.shape)}"
            )
        latent_dist, residual_dist = self._distributions(observations)
        batch_size = observations.shape[0]
        latent_log_prob = latent_dist.log_prob(latent_raw).reshape(
            batch_size, self.horizon, self.action_dim
        )
        residual_log_prob = residual_dist.log_prob(residual_raw).reshape(
            batch_size, self.horizon, self.action_dim
        )
        log_prob_steps = latent_log_prob.sum(-1) + residual_log_prob.sum(-1)
        log_prob = log_prob_steps.sum(-1)
        entropy = latent_dist.entropy().sum(-1) + residual_dist.entropy().sum(-1)
        chunk_shape = (observations.shape[0], self.horizon, self.action_dim)
        return RFSAction(
            latent=latent_raw.reshape(chunk_shape),
            residual=(self.residual_scale * torch.tanh(residual_raw)).reshape(chunk_shape),
            latent_raw=latent_raw,
            residual_raw=residual_raw,
            log_prob=log_prob,
            log_prob_steps=log_prob_steps,
            entropy=entropy,
        )


def clipped_ppo_loss(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    clip_coef: float,
) -> tuple[Tensor, Tensor]:
    if clip_coef <= 0:
        raise ValueError("clip_coef must be positive")
    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.clamp(-20.0, 20.0).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
    return loss, clip_fraction


def temporal_clipped_ppo_loss(
    new_log_prob_steps: Tensor,
    old_log_prob_steps: Tensor,
    advantages: Tensor,
    clip_coef: float,
) -> tuple[Tensor, Tensor]:
    if new_log_prob_steps.shape != old_log_prob_steps.shape:
        raise ValueError("new and old temporal log probabilities must have matching shapes")
    if new_log_prob_steps.ndim != 2 or advantages.shape != new_log_prob_steps.shape[:1]:
        raise ValueError("temporal PPO expects (batch, horizon) ratios and (batch,) advantages")
    if clip_coef <= 0:
        raise ValueError("clip_coef must be positive")
    log_ratio = new_log_prob_steps - old_log_prob_steps
    ratio = log_ratio.clamp(-20.0, 20.0).exp()
    weighted = ratio * advantages[:, None]
    clipped = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages[:, None]
    loss = -torch.minimum(weighted, clipped).sum(-1).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
    return loss, clip_fraction


@torch.no_grad()
def assert_frozen_parameters_unchanged(
    module: nn.Module,
    reference: dict[str, Tensor],
) -> None:
    current = module.state_dict()
    if current.keys() != reference.keys():
        raise RuntimeError("frozen base-policy state keys changed")
    changed = [name for name in current if not torch.equal(current[name].cpu(), reference[name])]
    if changed:
        raise RuntimeError(f"frozen base-policy parameters changed: {changed[:3]}")
