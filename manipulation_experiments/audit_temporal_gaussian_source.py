#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from eval_checkpoint import load_policy
from src.dexmg_env import create_vectorized_env
from src.source_priors import update_ar1_gaussian_source


@dataclass
class AuditConfig:
    checkpoint: Path
    output_json: Path
    device: str = "cuda"
    seed: int = 20260830
    correlation: float = 0.9
    num_envs: int = 16
    observation_batches: int = 4
    sequence_steps: int = 33
    sampling_steps: int = 10
    scalar_samples: int = 16384


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    return float(
        (centered_x * centered_y).mean()
        / (centered_x.square().mean().sqrt() * centered_y.square().mean().sqrt())
    )


def main(cfg: AuditConfig) -> None:
    torch.manual_seed(cfg.seed)
    policy = load_policy(cfg.checkpoint, device=cfg.device, load_ema=True)
    policy.config.sampling_steps = cfg.sampling_steps

    image_keys = [
        key.replace("observation.images.", "") for key in policy.config.image_features
    ]
    env = create_vectorized_env(
        env_name="Can",
        num_envs=cfg.num_envs,
        device="cuda" if cfg.device != "cpu" else "cpu",
        camera_size=84,
        render_size=(240, 320),
        video_key="agentview",
        debug=False,
        expected_image_keys=image_keys,
    )
    observation_batches = []
    for _ in range(cfg.observation_batches):
        observations, _ = env.reset()
        observation_batches.append(observations)
    env.close()
    fixed_observations = {
        key: torch.cat([batch[key] for batch in observation_batches], dim=0)
        for key in observation_batches[0]
    }

    batch_size = next(iter(fixed_observations.values())).shape[0]
    shape = (batch_size, policy.config.horizon, policy.model.action_dim)
    iid_actions = []
    correlated_actions = []
    previous_source = None
    with torch.inference_mode():
        for _ in range(cfg.sequence_steps):
            innovation = torch.randn(shape, device=cfg.device)
            correlated_source = update_ar1_gaussian_source(
                innovation, previous_source, cfg.correlation
            )
            previous_source = correlated_source
            iid_chunk, _ = policy.predict_action_chunk(
                dict(fixed_observations), source_noise=innovation
            )
            correlated_chunk, _ = policy.predict_action_chunk(
                dict(fixed_observations), source_noise=correlated_source
            )
            iid_actions.append(iid_chunk.cpu())
            correlated_actions.append(correlated_chunk.cpu())

    iid_actions_tensor = torch.stack(iid_actions)
    correlated_actions_tensor = torch.stack(correlated_actions)
    iid_change = torch.linalg.vector_norm(
        iid_actions_tensor[1:] - iid_actions_tensor[:-1], dim=(-2, -1)
    ).mean()
    correlated_change = torch.linalg.vector_norm(
        correlated_actions_tensor[1:] - correlated_actions_tensor[:-1],
        dim=(-2, -1),
    ).mean()
    iid_diversity = iid_actions_tensor.std(dim=0, unbiased=False).mean()
    correlated_diversity = correlated_actions_tensor.std(
        dim=0, unbiased=False
    ).mean()

    previous = torch.randn(cfg.scalar_samples)
    innovation = torch.randn(cfg.scalar_samples)
    current = update_ar1_gaussian_source(
        innovation, previous, cfg.correlation
    )
    results = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
        "source": {
            "mean": float(current.mean()),
            "std": float(current.std(unbiased=False)),
            "lag_one_correlation": _correlation(previous, current),
            "finite": bool(torch.isfinite(current).all()),
        },
        "actions": {
            "fixed_observations": batch_size,
            "transitions_per_observation": cfg.sequence_steps - 1,
            "iid_mean_chunk_change_l2": float(iid_change),
            "correlated_mean_chunk_change_l2": float(correlated_change),
            "relative_chunk_change": float(correlated_change / iid_change),
            "iid_diversity": float(iid_diversity),
            "correlated_diversity": float(correlated_diversity),
            "diversity_retention": float(correlated_diversity / iid_diversity),
            "finite": bool(
                torch.isfinite(iid_actions_tensor).all()
                and torch.isfinite(correlated_actions_tensor).all()
            ),
        },
    }
    cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_json.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(tyro.cli(AuditConfig))
