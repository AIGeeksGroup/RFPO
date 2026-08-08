#!/usr/bin/env python

"""Evaluate conditional flow geometry and low-NFE endpoint error on fixed pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from safetensors.torch import load_file

from src.flow_metrics import flow_geometry
from src.flow_model import FlowMatchingPolicy
from src.flow_model_config import FlowMatchingConfig


@dataclass
class GeometryConfig:
    checkpoint_path: str
    output_json: str
    dataset: str = "ankile/robomimic-mh-can-image"
    sampling_steps: str = "64,10,4,2,1"
    reference_steps: int = 64
    batch_size: int = 64
    num_batches: int = 2
    diversity_samples: int = 64
    seed: int = 20260808
    num_workers: int = 2
    device: str = "cuda"
    load_ema: bool = True


def _load_policy(cfg: GeometryConfig, metadata: LeRobotDatasetMetadata) -> FlowMatchingPolicy:
    checkpoint_path = Path(cfg.checkpoint_path)
    config_dict = json.loads((checkpoint_path / "policy" / "config.json").read_text())
    config_dict.pop("type", None)
    config_dict.pop("normalization_mapping", None)
    policy_cfg = FlowMatchingConfig(**config_dict)
    policy = FlowMatchingPolicy(policy_cfg, dataset_stats=metadata.stats)
    policy.load_state_dict(load_file(checkpoint_path / "policy" / "model.safetensors", device="cpu"))

    optimizer_state = torch.load(checkpoint_path / "optimizer.pt", map_location="cpu", weights_only=False)
    if cfg.load_ema:
        if policy.ema_model is None or "ema_state_dict" not in optimizer_state:
            raise ValueError("EMA weights requested but unavailable")
        policy.ema_model.load_state_dict(optimizer_state["ema_state_dict"])
        policy.ema_model.copy_to(policy.model.parameters())

    policy.to(cfg.device)
    if policy.ema_model is not None:
        policy.ema_model.to(cfg.device)
    policy.eval()
    policy.requires_grad_(False)
    return policy


def _prepare_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    prepared = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared[key] = value.to(device, non_blocking=True)
        else:
            prepared[key] = value
    if "state" in prepared and "observation.state" not in prepared:
        prepared["observation.state"] = prepared.pop("state")
    if "actions" in prepared and "action" not in prepared:
        prepared["action"] = prepared.pop("actions")
    return prepared


def _repeat_observations(batch: dict[str, Any], count: int) -> dict[str, Any]:
    repeated = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[0] > 0:
            repeated[key] = value[:1].expand(count, *value.shape[1:]).clone()
    return repeated


@torch.no_grad()
def main(cfg: GeometryConfig) -> None:
    if cfg.reference_steps < 1 or cfg.num_batches < 1 or cfg.batch_size < 1:
        raise ValueError("reference_steps, num_batches, and batch_size must be positive")
    steps = sorted({int(value) for value in cfg.sampling_steps.split(",")}, reverse=True)
    if any(value < 1 for value in steps):
        raise ValueError("all sampling steps must be positive")
    if cfg.reference_steps not in steps:
        steps.insert(0, cfg.reference_steps)

    torch.manual_seed(cfg.seed)
    metadata = LeRobotDatasetMetadata(cfg.dataset)
    policy = _load_policy(cfg, metadata)
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = LeRobotDataset(cfg.dataset, delta_timestamps=delta_timestamps, download_videos=True)
    loader_generator = torch.Generator().manual_seed(cfg.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        drop_last=True,
    )
    source_generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

    aggregate = {
        step: {"count": 0, "straightness_error": 0.0, "path_length_ratio": 0.0, "endpoint_mse": 0.0}
        for step in steps
    }
    diversity_batch = None

    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= cfg.num_batches:
            break
        batch = _prepare_batch(raw_batch, cfg.device)
        if diversity_batch is None:
            diversity_batch = _repeat_observations(batch, cfg.diversity_samples)
        source = torch.randn(
            (cfg.batch_size, policy.config.horizon, policy.model.action_dim),
            generator=source_generator,
            device=cfg.device,
        )

        policy.config.sampling_steps = cfg.reference_steps
        reference_actions, _ = policy.predict_action_chunk(batch, source_noise=source)

        for step in steps:
            policy.config.sampling_steps = step
            actions, path = policy.predict_action_chunk(batch, source_noise=source)
            geometry = flow_geometry(source, path)
            count = source.shape[0]
            values = aggregate[step]
            values["count"] += count
            values["straightness_error"] += geometry["straightness_error"].sum().item()
            values["path_length_ratio"] += geometry["path_length_ratio"].sum().item()
            values["endpoint_mse"] += (actions - reference_actions).square().flatten(1).mean(dim=1).sum().item()

    if diversity_batch is None:
        raise RuntimeError("dataset yielded no full batches")

    diversity_source = torch.randn(
        (cfg.diversity_samples, policy.config.horizon, policy.model.action_dim),
        generator=source_generator,
        device=cfg.device,
    )
    diversity = {}
    for step in steps:
        policy.config.sampling_steps = step
        actions, _ = policy.predict_action_chunk(diversity_batch, source_noise=diversity_source)
        flattened = actions.flatten(1)
        diversity[str(step)] = {
            "mean_element_std": flattened.std(dim=0).mean().item(),
            "mean_pairwise_distance": torch.pdist(flattened).mean().item(),
        }

    metrics = {}
    for step, values in aggregate.items():
        count = values.pop("count")
        metrics[str(step)] = {key: value / count for key, value in values.items()}

    result = {
        "checkpoint_path": str(Path(cfg.checkpoint_path).resolve()),
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "num_pairs": cfg.batch_size * cfg.num_batches,
        "reference_steps": cfg.reference_steps,
        "metrics": metrics,
        "source_conditioned_diversity": diversity,
    }
    output_path = Path(cfg.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(GeometryConfig))

