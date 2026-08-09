#!/usr/bin/env python

from __future__ import annotations

import copy
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor, nn, optim
import tyro

from src.dexmg_env import create_vectorized_env
from src.flow_model import FlowMatchingPolicy
from src.flow_model_config import FlowMatchingConfig
from src.residual_flow_steering import (
    ResidualFlowSteeringPolicy,
    assert_frozen_parameters_unchanged,
    clipped_ppo_loss,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    mode: Literal["train", "eval"] = "train"
    base_policy_local_path: str = ""
    rfs_checkpoint: str | None = None
    output_dir: str = "runs/rfs"
    task: str = "Square"
    device: str = "cuda"
    seed: int = 20260924
    num_envs: int = 16
    camera_size: int = 84
    sampling_steps: int = 10
    n_action_steps: int = 16
    collection_steps: int = 320
    total_timesteps: int = 25_600
    update_epochs: int = 10
    num_minibatches: int = 4
    learning_rate_policy: float = 3e-4
    learning_rate_value: float = 1e-3
    discount: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    residual_scale: float = 0.1
    eval_episodes: int = 20
    deterministic: bool = False


class ValueNetwork(nn.Module):
    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.network(observations).squeeze(-1)


def load_base_policy(path: str, device: torch.device, sampling_steps: int, n_action_steps: int):
    policy_dir = Path(path)
    if (policy_dir / "policy").is_dir():
        policy_dir = policy_dir / "policy"
    with (policy_dir / "config.json").open() as handle:
        config_payload = json.load(handle)
    config_payload.pop("type", None)
    config_payload.pop("normalization_mapping", None)
    policy_config = FlowMatchingConfig(**config_payload)
    policy_config.image_features = [key for key in policy_config.input_features if "image" in key]
    policy_config.state_features = [
        key for key in policy_config.input_features if "state" in key or "pos" in key
    ]
    policy_config.sampling_steps = sampling_steps
    policy_config.n_action_steps = n_action_steps
    actor = FlowMatchingPolicy(policy_config, dataset_stats=None)
    actor.load_state_dict(load_file(policy_dir / "model.safetensors", device="cpu"), strict=True)
    optimizer_path = policy_dir.parent / "optimizer.pt"
    if optimizer_path.exists():
        optimizer_blob = torch.load(optimizer_path, map_location="cpu")
        ema_state = optimizer_blob.get("ema_state_dict")
        if ema_state is not None and getattr(actor, "ema_model", None) is not None:
            actor.ema_model.load_state_dict(ema_state)
            actor.ema_model.copy_to(actor.model.parameters())
    actor.to(device).eval().requires_grad_(False)
    return actor


@torch.no_grad()
def encode(actor: FlowMatchingPolicy, observations: dict[str, Tensor]) -> Tensor:
    normalized = actor.normalize_inputs(copy.deepcopy(observations))
    return actor.model.encode_observations(normalized)


def subset_observations(observations: dict[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    return {key: value[indices] for key, value in observations.items()}


def make_environment(cfg: Config, actor: FlowMatchingPolicy):
    image_keys = [key.replace("observation.images.", "") for key in actor.config.image_features]
    return create_vectorized_env(
        env_name=cfg.task,
        num_envs=cfg.num_envs,
        device="cuda" if torch.device(cfg.device).type == "cuda" else "cpu",
        camera_size=cfg.camera_size,
        video_key="agentview",
        expected_image_keys=image_keys,
        seeds=[cfg.seed + env_id for env_id in range(cfg.num_envs)],
    )


def action_bounds(env, device: torch.device) -> tuple[Tensor, Tensor]:
    low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    return low, high


def finish_transition(active: dict, env_id: int, done: bool) -> dict:
    duration = active["duration"][env_id]
    transition = {
        "observation": active["observation"][env_id].cpu(),
        "latent_raw": active["latent_raw"][env_id].cpu(),
        "residual_raw": active["residual_raw"][env_id].cpu(),
        "old_log_prob": active["old_log_prob"][env_id].cpu(),
        "value": active["value"][env_id].cpu(),
        "reward": active["reward"][env_id].cpu(),
        "duration": duration,
        "done": done,
    }
    active["valid"][env_id] = False
    return transition


@torch.no_grad()
def start_plans(
    actor: FlowMatchingPolicy,
    policy: ResidualFlowSteeringPolicy,
    value: ValueNetwork,
    observations: dict[str, Tensor],
    env_ids: Tensor,
    active: dict,
    deterministic: bool,
    low: Tensor,
    high: Tensor,
) -> None:
    if env_ids.numel() == 0:
        return
    sub_obs = subset_observations(observations, env_ids)
    conditioning = encode(actor, sub_obs)
    modulation = policy.sample(conditioning, deterministic=deterministic)
    base_actions, _ = actor.predict_action_chunk(sub_obs, source_noise=modulation.latent)
    actions = (base_actions + modulation.residual).clamp(low, high)
    for local_id, env_tensor in enumerate(env_ids):
        env_id = int(env_tensor.item())
        active["valid"][env_id] = True
        active["observation"][env_id] = conditioning[local_id]
        active["latent_raw"][env_id] = modulation.latent_raw[local_id]
        active["residual_raw"][env_id] = modulation.residual_raw[local_id]
        active["old_log_prob"][env_id] = modulation.log_prob[local_id]
        active["value"][env_id] = value(conditioning[local_id : local_id + 1])[0]
        active["actions"][env_id] = actions[local_id]
        active["reward"][env_id] = torch.zeros((), device=conditioning.device)
        active["duration"][env_id] = 0


def empty_active(num_envs: int) -> dict:
    return {
        "valid": torch.zeros(num_envs, dtype=torch.bool),
        "observation": [None] * num_envs,
        "latent_raw": [None] * num_envs,
        "residual_raw": [None] * num_envs,
        "old_log_prob": [None] * num_envs,
        "value": [None] * num_envs,
        "actions": [None] * num_envs,
        "reward": [None] * num_envs,
        "duration": [0] * num_envs,
    }


@torch.no_grad()
def collect_rollout(cfg, env, actor, policy, value, observations, low, high):
    active = empty_active(cfg.num_envs)
    trajectories: list[list[dict]] = [[] for _ in range(cfg.num_envs)]
    successes = episodes = 0
    for _ in range(cfg.collection_steps):
        inactive = torch.where(~active["valid"])[0].to(torch.device(cfg.device))
        start_plans(actor, policy, value, observations, inactive, active, False, low, high)
        actions = torch.stack(
            [active["actions"][i][active["duration"][i]] for i in range(cfg.num_envs)]
        )
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        for env_id in range(cfg.num_envs):
            duration = active["duration"][env_id]
            active["reward"][env_id] += (cfg.discount**duration) * rewards[env_id]
            active["duration"][env_id] += 1
            if dones[env_id] or active["duration"][env_id] == cfg.n_action_steps:
                trajectories[env_id].append(
                    finish_transition(active, env_id, bool(dones[env_id].item()))
                )
            if dones[env_id]:
                episodes += 1
                successes += int(rewards[env_id].item() == 1.0)
        observations = next_observations

    for env_id in range(cfg.num_envs):
        if active["valid"][env_id]:
            trajectories[env_id].append(finish_transition(active, env_id, False))
    next_conditioning = encode(actor, observations)
    next_values = value(next_conditioning).cpu()
    return trajectories, observations, next_values, successes, episodes


def flatten_with_advantages(cfg, trajectories, next_values):
    flattened = []
    for env_id, trajectory in enumerate(trajectories):
        next_advantage = torch.zeros(())
        bootstrap = next_values[env_id]
        for index in reversed(range(len(trajectory))):
            item = trajectory[index]
            nonterminal = 0.0 if item["done"] else 1.0
            following_value = bootstrap if index == len(trajectory) - 1 else trajectory[index + 1]["value"]
            gamma_duration = cfg.discount ** item["duration"]
            trace_duration = (cfg.discount * cfg.gae_lambda) ** item["duration"]
            delta = item["reward"] + gamma_duration * nonterminal * following_value - item["value"]
            item["advantage"] = delta + trace_duration * nonterminal * next_advantage
            item["return"] = item["advantage"] + item["value"]
            next_advantage = item["advantage"]
        flattened.extend(trajectory)
    return flattened


def stack_batch(transitions: list[dict], device: torch.device) -> dict[str, Tensor]:
    keys = (
        "observation",
        "latent_raw",
        "residual_raw",
        "old_log_prob",
        "advantage",
        "return",
    )
    return {key: torch.stack([item[key] for item in transitions]).to(device) for key in keys}


def ppo_update(cfg, policy, value, policy_optimizer, value_optimizer, batch):
    size = batch["advantage"].numel()
    advantages = batch["advantage"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    minibatch_size = max(size // cfg.num_minibatches, 1)
    metrics = []
    for _ in range(cfg.update_epochs):
        for indices in torch.randperm(size, device=advantages.device).split(minibatch_size):
            evaluated = policy.evaluate_actions(
                batch["observation"][indices],
                batch["latent_raw"][indices],
                batch["residual_raw"][indices],
            )
            policy_loss, clip_fraction = clipped_ppo_loss(
                evaluated.log_prob,
                batch["old_log_prob"][indices],
                advantages[indices],
                cfg.clip_coef,
            )
            policy_objective = policy_loss - cfg.entropy_coef * evaluated.entropy.mean()
            value_loss = 0.5 * (
                value(batch["observation"][indices]) - batch["return"][indices]
            ).square().mean()

            policy_optimizer.zero_grad(set_to_none=True)
            policy_objective.backward()
            policy_grad = nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            policy_optimizer.step()

            value_optimizer.zero_grad(set_to_none=True)
            (cfg.value_loss_coef * value_loss).backward()
            value_grad = nn.utils.clip_grad_norm_(value.parameters(), cfg.max_grad_norm)
            value_optimizer.step()
            metrics.append(
                (policy_loss.item(), value_loss.item(), clip_fraction.item(), float(policy_grad), float(value_grad))
            )
    return np.asarray(metrics).mean(axis=0)


def save_rfs_checkpoint(path: Path, cfg: Config, policy, value, iteration: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "iteration": iteration,
        },
        path,
    )


def build_rfs(cfg: Config, actor: FlowMatchingPolicy, device: torch.device):
    policy = ResidualFlowSteeringPolicy(
        actor.model.global_cond_dim,
        actor.config.horizon,
        actor.model.action_dim,
        residual_scale=cfg.residual_scale,
    ).to(device)
    value = ValueNetwork(actor.model.global_cond_dim).to(device)
    if cfg.rfs_checkpoint is not None:
        payload = torch.load(cfg.rfs_checkpoint, map_location=device)
        policy.load_state_dict(payload["policy"])
        value.load_state_dict(payload["value"])
    return policy, value


def train(cfg: Config, env, actor, policy, value, device, output_dir: Path) -> None:
    base_reference = {name: tensor.detach().cpu().clone() for name, tensor in actor.state_dict().items()}
    latent_head_reference = policy.mean_head.weight[: policy.chunk_dim].detach().clone()
    residual_head_reference = policy.mean_head.weight[policy.chunk_dim :].detach().clone()
    policy_optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate_policy)
    value_optimizer = optim.Adam(value.parameters(), lr=cfg.learning_rate_value)
    observations, _ = env.reset()
    low, high = action_bounds(env, device)
    steps_per_iteration = cfg.collection_steps * cfg.num_envs
    iterations = cfg.total_timesteps // steps_per_iteration
    if iterations < 1 or cfg.total_timesteps % steps_per_iteration:
        raise ValueError("total_timesteps must be a positive multiple of collection_steps * num_envs")

    history = []
    for iteration in range(1, iterations + 1):
        trajectories, observations, next_values, successes, episodes = collect_rollout(
            cfg, env, actor, policy, value, observations, low, high
        )
        transitions = flatten_with_advantages(cfg, trajectories, next_values)
        batch = stack_batch(transitions, device)
        metrics = ppo_update(
            cfg, policy, value, policy_optimizer, value_optimizer, batch
        )
        assert_frozen_parameters_unchanged(actor, base_reference)
        latent_head_delta = (
            policy.mean_head.weight[: policy.chunk_dim] - latent_head_reference
        ).norm()
        residual_head_delta = (
            policy.mean_head.weight[policy.chunk_dim :] - residual_head_reference
        ).norm()
        if min(latent_head_delta.item(), residual_head_delta.item()) <= 0:
            raise RuntimeError("RFS PPO failed to update both modulation branches")
        record = {
            "iteration": iteration,
            "transitions": len(transitions),
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / max(episodes, 1),
            "policy_loss": float(metrics[0]),
            "value_loss": float(metrics[1]),
            "clip_fraction": float(metrics[2]),
            "policy_grad_norm": float(metrics[3]),
            "value_grad_norm": float(metrics[4]),
            "latent_log_std": float(policy.latent_log_std.mean().item()),
            "residual_log_std": float(policy.residual_log_std.mean().item()),
            "latent_head_delta": float(latent_head_delta.item()),
            "residual_head_delta": float(residual_head_delta.item()),
        }
        if not all(np.isfinite(value) for value in record.values() if isinstance(value, float)):
            raise RuntimeError(f"non-finite RFS metrics: {record}")
        history.append(record)
        logger.info("RFS iteration %s", json.dumps(record, sort_keys=True))
        save_rfs_checkpoint(output_dir / "latest.pt", cfg, policy, value, iteration)
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")


@torch.no_grad()
def evaluate(cfg: Config, env, actor, policy, value, device, output_dir: Path) -> None:
    observations, _ = env.reset()
    low, high = action_bounds(env, device)
    active = empty_active(cfg.num_envs)
    complete = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
    successes = torch.zeros(cfg.num_envs, dtype=torch.int64, device=device)
    episode_steps = torch.zeros(cfg.num_envs, dtype=torch.int64, device=device)
    while not bool(complete.all()):
        inactive = torch.where(~active["valid"])[0].to(device)
        start_plans(actor, policy, value, observations, inactive, active, cfg.deterministic, low, high)
        actions = torch.stack(
            [active["actions"][i][active["duration"][i]] for i in range(cfg.num_envs)]
        )
        observations, rewards, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        episode_steps[~complete] += 1
        for env_id in range(cfg.num_envs):
            active["duration"][env_id] += 1
            if dones[env_id] or active["duration"][env_id] == cfg.n_action_steps:
                active["valid"][env_id] = False
            if dones[env_id] and not complete[env_id]:
                complete[env_id] = True
                successes[env_id] = int(rewards[env_id].item() == 1.0)
    result = {
        "deterministic": cfg.deterministic,
        "successes": int(successes.sum().item()),
        "episodes": cfg.num_envs,
        "success_rate": float(successes.float().mean().item()),
        "episode_steps": episode_steps.cpu().tolist(),
    }
    logger.info("RFS evaluation %s", json.dumps(result, sort_keys=True))
    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")


def main(cfg: Config) -> None:
    if not cfg.base_policy_local_path:
        raise ValueError("base_policy_local_path is required")
    if cfg.mode == "eval" and cfg.rfs_checkpoint is None:
        raise ValueError("rfs_checkpoint is required for evaluation")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    actor = load_base_policy(
        cfg.base_policy_local_path, device, cfg.sampling_steps, cfg.n_action_steps
    )
    policy, value = build_rfs(cfg, actor, device)
    env = make_environment(cfg, actor)
    try:
        if cfg.mode == "train":
            train(cfg, env, actor, policy, value, device, output_dir)
        else:
            evaluate(cfg, env, actor, policy, value, device, output_dir)
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Config, config=(tyro.conf.FlagConversionOff,)))
