"""Continuous-action centering and chunk-level DAE utilities."""

from __future__ import annotations

import torch
from torch import nn


class DirectAdvantageHead(nn.Module):
    def __init__(self, observation_dim: int, action_size: int):
        super().__init__()
        if observation_dim <= 0 or action_size <= 0:
            raise ValueError("observation_dim and action_size must be positive")
        self.observation_dim = observation_dim
        self.action_size = action_size
        self.mlp = nn.Sequential(
            nn.Linear(observation_dim + action_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def raw(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError("observations must have shape [batch, observation_dim]")
        if actions.shape[0] != observations.shape[0]:
            raise ValueError("observations and actions must have matching batch dimensions")
        flat_actions = actions.reshape(actions.shape[0], -1)
        if flat_actions.shape[1] != self.action_size:
            raise ValueError("actions do not match the configured flattened action size")
        return self.mlp(torch.cat([observations, flat_actions], dim=1)).squeeze(1)

    def forward(
        self,
        observations: torch.Tensor,
        behavior_actions: torch.Tensor,
        centering_actions: torch.Tensor,
    ) -> torch.Tensor:
        if centering_actions.ndim < 3:
            raise ValueError("centering_actions must have shape [batch, samples, ...]")
        batch_size, samples = centering_actions.shape[:2]
        if batch_size != observations.shape[0] or samples <= 0:
            raise ValueError("centering actions must provide samples for every observation")
        behavior_values = self.raw(observations, behavior_actions)
        repeated_observations = observations[:, None].expand(
            batch_size, samples, self.observation_dim
        )
        center_values = self.raw(
            repeated_observations.reshape(-1, self.observation_dim),
            centering_actions.reshape(batch_size * samples, *centering_actions.shape[2:]),
        ).reshape(batch_size, samples)
        return behavior_values - center_values.mean(dim=1)


def discounted_macro_rewards(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    action_steps: int,
    discount: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate primitive rewards into action-chunk rewards without crossing terminals."""
    if rewards.ndim != 2 or terminals.shape != rewards.shape:
        raise ValueError("rewards and terminals must have matching [time, env] shapes")
    if action_steps <= 0 or rewards.shape[0] % action_steps:
        raise ValueError("action_steps must evenly divide the rollout length")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0, 1]")

    chunks = rewards.shape[0] // action_steps
    chunk_rewards = torch.zeros(
        (chunks, rewards.shape[1]), dtype=rewards.dtype, device=rewards.device
    )
    chunk_terminals = torch.zeros(
        (chunks, rewards.shape[1]), dtype=torch.bool, device=rewards.device
    )
    for chunk in range(chunks):
        alive = torch.ones(rewards.shape[1], dtype=torch.bool, device=rewards.device)
        for offset in range(action_steps):
            index = chunk * action_steps + offset
            chunk_rewards[chunk] += (
                (discount**offset) * rewards[index] * alive.to(rewards.dtype)
            )
            newly_terminal = terminals[index].bool() & alive
            chunk_terminals[chunk] |= newly_terminal
            alive &= ~newly_terminal
    return chunk_rewards, chunk_terminals


def direct_advantage_residuals(
    advantages: torch.Tensor,
    macro_rewards: torch.Tensor,
    terminals: torch.Tensor,
    baseline_values: torch.Tensor,
    final_baseline_values: torch.Tensor,
    valid_chunks: torch.Tensor,
    macro_discount: float,
    horizon: int,
) -> torch.Tensor:
    """Build fixed-baseline DAE residuals for valid macro-action windows."""
    expected_shape = advantages.shape
    if advantages.ndim != 2:
        raise ValueError("chunk tensors must have shape [chunks, env]")
    if any(
        tensor.shape != expected_shape
        for tensor in (macro_rewards, terminals, baseline_values, valid_chunks)
    ):
        raise ValueError("all chunk tensors must have matching shapes")
    if final_baseline_values.shape != (advantages.shape[1],):
        raise ValueError("final_baseline_values must have shape [env]")
    if not 0.0 <= macro_discount <= 1.0 or horizon <= 0:
        raise ValueError("macro_discount and horizon are invalid")

    residuals: list[torch.Tensor] = []
    chunks, environments = advantages.shape
    valid_chunks = valid_chunks.bool()
    terminals = terminals.bool()
    for start in range(chunks):
        for environment in range(environments):
            if not valid_chunks[start, environment]:
                continue
            residual = -baseline_values[start, environment]
            multiplier = 1.0
            complete = False
            for offset in range(horizon):
                index = start + offset
                if index >= chunks or not valid_chunks[index, environment]:
                    break
                residual = residual + multiplier * (
                    macro_rewards[index, environment] - advantages[index, environment]
                )
                multiplier *= macro_discount
                if terminals[index, environment]:
                    complete = True
                    break
            else:
                next_index = start + horizon
                if next_index < chunks and valid_chunks[next_index, environment]:
                    residual = residual + multiplier * baseline_values[next_index, environment]
                    complete = True
                elif next_index == chunks:
                    residual = residual + multiplier * final_baseline_values[environment]
                    complete = True
            if complete:
                residuals.append(residual)
    if not residuals:
        raise ValueError("no complete valid DAE windows are available")
    return torch.stack(residuals)
