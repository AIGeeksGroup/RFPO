#!/usr/bin/env python

"""Measure disagreement among fixed Monte Carlo CFM gradients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import tyro
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from safetensors.torch import load_file

from src.flow_model import FlowMatchingPolicy
from src.flow_model_config import FlowMatchingConfig


@dataclass
class GradientAuditConfig:
    checkpoint_path: str
    output_json: str
    dataset: str = "ankile/robomimic-mh-can-image"
    batch_size: int = 16
    num_batches: int = 2
    num_cfm_samples: int = 8
    seed: int = 20260808
    num_workers: int = 2
    device: str = "cuda"
    load_ema: bool = True


def _load_policy(
    cfg: GradientAuditConfig, metadata: LeRobotDatasetMetadata
) -> FlowMatchingPolicy:
    checkpoint_path = Path(cfg.checkpoint_path)
    config_dict = json.loads((checkpoint_path / "policy" / "config.json").read_text())
    config_dict.pop("type", None)
    config_dict.pop("normalization_mapping", None)
    policy_cfg = FlowMatchingConfig(**config_dict)
    policy_cfg.image_features = [key for key in policy_cfg.input_features if "image" in key]
    policy_cfg.state_features = [
        key for key in policy_cfg.input_features if "state" in key or "pos" in key
    ]
    policy_cfg.cfm_loss_use_huber = True
    policy_cfg.cfm_loss_huber_delta = 0.5
    policy = FlowMatchingPolicy(policy_cfg, dataset_stats=metadata.stats)
    policy.load_state_dict(
        load_file(checkpoint_path / "policy" / "model.safetensors", device="cpu")
    )

    optimizer_state = torch.load(
        checkpoint_path / "optimizer.pt", map_location="cpu", weights_only=False
    )
    if cfg.load_ema:
        if policy.ema_model is None or "ema_state_dict" not in optimizer_state:
            raise ValueError("EMA weights requested but unavailable")
        policy.ema_model.load_state_dict(optimizer_state["ema_state_dict"])
        policy.ema_model.copy_to(policy.model.parameters())

    policy.to(cfg.device)
    policy.eval()
    policy.requires_grad_(False)
    for name, parameter in policy.model.named_parameters():
        if "vision_encoder" not in name:
            parameter.requires_grad_(True)
    return policy


def _prepare_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    prepared = {}
    for key, value in batch.items():
        prepared[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    if "state" in prepared and "observation.state" not in prepared:
        prepared["observation.state"] = prepared.pop("state")
    if "actions" in prepared and "action" not in prepared:
        prepared["action"] = prepared.pop("actions")
    return prepared


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _gradient_vector(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter]
) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    pieces = [
        (torch.zeros_like(parameter) if gradient is None else gradient).detach().flatten().cpu()
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]
    return torch.cat(pieces)


def _coefficient_of_variation(values: torch.Tensor) -> float:
    return (values.std(unbiased=False) / values.mean().abs().clamp_min(1e-12)).item()


def main(cfg: GradientAuditConfig) -> None:
    if cfg.batch_size < 1 or cfg.num_batches < 1 or cfg.num_cfm_samples < 2:
        raise ValueError("batch_size and num_batches must be positive; num_cfm_samples must exceed 1")

    torch.manual_seed(cfg.seed)
    metadata = LeRobotDatasetMetadata(cfg.dataset)
    policy = _load_policy(cfg, metadata)
    parameters = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no non-vision actor parameters selected")

    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = LeRobotDataset(cfg.dataset, delta_timestamps=delta_timestamps, download_videos=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        drop_last=True,
    )
    cfm_generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 1)
    batch_metrics = []

    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= cfg.num_batches:
            break
        batch = _prepare_batch(raw_batch, cfg.device)
        actions = batch["action"]
        losses = []
        gradient_vectors = []

        for _ in range(cfg.num_cfm_samples):
            cfm_t = torch.rand(
                (cfg.batch_size, 1, 1), generator=cfm_generator, device=cfg.device
            )
            cfm_epsilon = torch.randn(
                actions.shape, generator=cfm_generator, device=cfg.device
            )
            cfm_loss, _, _ = policy(
                _clone_batch(batch),
                n_action_samples=1,
                cfm_loss_t=cfm_t,
                cfm_loss_eps=cfm_epsilon,
            )
            loss = cfm_loss.mean()
            losses.append(loss.detach().cpu())
            gradient_vectors.append(_gradient_vector(loss, parameters))

        loss_values = torch.stack(losses)
        gradients = torch.stack(gradient_vectors)
        if not torch.isfinite(gradients).all():
            raise RuntimeError(f"non-finite gradient in batch {batch_index}")
        gradient_norms = gradients.norm(dim=1)
        if (gradient_norms == 0).any():
            raise RuntimeError(f"zero gradient in batch {batch_index}")
        average_gradient = gradients.mean(dim=0)
        cosine = F.cosine_similarity(
            gradients, average_gradient.unsqueeze(0).expand_as(gradients), dim=1
        )
        batch_metrics.append(
            {
                "batch": batch_index,
                "mean_gradient_cosine": cosine.mean().item(),
                "std_gradient_cosine": cosine.std(unbiased=False).item(),
                "gradient_norm_mean": gradient_norms.mean().item(),
                "gradient_norm_cv": _coefficient_of_variation(gradient_norms),
                "cfm_loss_mean": loss_values.mean().item(),
                "cfm_loss_cv": _coefficient_of_variation(loss_values),
            }
        )

    if len(batch_metrics) != cfg.num_batches:
        raise RuntimeError(f"expected {cfg.num_batches} batches, received {len(batch_metrics)}")

    aggregate_keys = [key for key in batch_metrics[0] if key != "batch"]
    aggregate = {
        key: sum(batch[key] for batch in batch_metrics) / len(batch_metrics)
        for key in aggregate_keys
    }
    result = {
        "checkpoint_path": str(Path(cfg.checkpoint_path).resolve()),
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "num_batches": cfg.num_batches,
        "num_cfm_samples": cfg.num_cfm_samples,
        "num_parameters": sum(parameter.numel() for parameter in parameters),
        "aggregate": aggregate,
        "batches": batch_metrics,
    }
    output_path = Path(cfg.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(GradientAuditConfig))
