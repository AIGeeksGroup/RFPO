#!/usr/bin/env python

"""Compare repeated iid and antithetic MC8 CFM gradient estimators."""

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

from src.cfm_sampling import sample_cfm_variables
from src.flow_model import FlowMatchingPolicy
from src.flow_model_config import FlowMatchingConfig


@dataclass
class EstimatorAuditConfig:
    checkpoint_path: str
    output_json: str
    dataset: str = "ankile/robomimic-mh-can-image"
    batch_size: int = 16
    num_batches: int = 2
    num_cfm_samples: int = 8
    num_repeats: int = 8
    seed: int = 20260813
    num_workers: int = 2
    device: str = "cuda"
    load_ema: bool = True


def _load_policy(
    cfg: EstimatorAuditConfig, metadata: LeRobotDatasetMetadata
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
    prepared = {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
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
    return torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient)
            .detach()
            .flatten()
            .cpu()
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    )


def _coefficient_of_variation(values: torch.Tensor) -> float:
    return (values.std(unbiased=False) / values.mean().abs().clamp_min(1e-12)).item()


def _audit_mode(
    *,
    mode: str,
    batch: dict[str, Any],
    policy: FlowMatchingPolicy,
    parameters: list[torch.nn.Parameter],
    cfg: EstimatorAuditConfig,
    batch_index: int,
) -> tuple[dict[str, float], torch.Tensor]:
    actions = batch["action"]
    time_generator = torch.Generator(device=cfg.device).manual_seed(
        cfg.seed + 1000 * batch_index + 1
    )
    noise_generator = torch.Generator(device=cfg.device).manual_seed(
        cfg.seed + 1000 * batch_index + 2
    )
    losses = []
    gradients = []
    for _ in range(cfg.num_repeats):
        cfm_t, cfm_epsilon = sample_cfm_variables(
            batch_size=cfg.batch_size,
            num_samples=cfg.num_cfm_samples,
            horizon=actions.shape[1],
            action_dim=actions.shape[2],
            mode=mode,
            time_generator=time_generator,
            noise_generator=noise_generator,
            device=cfg.device,
            dtype=actions.dtype,
        )
        cfm_loss, _, _ = policy(
            _clone_batch(batch),
            n_action_samples=cfg.num_cfm_samples,
            cfm_loss_t=cfm_t,
            cfm_loss_eps=cfm_epsilon,
        )
        loss = cfm_loss.mean()
        losses.append(loss.detach().cpu())
        gradients.append(_gradient_vector(loss, parameters))

    loss_values = torch.stack(losses)
    gradient_values = torch.stack(gradients)
    if not torch.isfinite(gradient_values).all():
        raise RuntimeError(f"non-finite {mode} gradient in batch {batch_index}")
    gradient_norms = gradient_values.norm(dim=1)
    if (gradient_norms == 0).any():
        raise RuntimeError(f"zero {mode} gradient in batch {batch_index}")
    average_gradient = gradient_values.mean(dim=0)
    deviations = gradient_values - average_gradient
    normalized_gradient_mse = (
        deviations.square().sum(dim=1).mean()
        / average_gradient.square().sum().clamp_min(1e-12)
    )
    cosine = F.cosine_similarity(
        gradient_values,
        average_gradient.unsqueeze(0).expand_as(gradient_values),
        dim=1,
    )
    metrics = {
        "loss_estimator_mean": loss_values.mean().item(),
        "loss_estimator_cv": _coefficient_of_variation(loss_values),
        "gradient_norm_mean": gradient_norms.mean().item(),
        "gradient_norm_cv": _coefficient_of_variation(gradient_norms),
        "mean_gradient_cosine": cosine.mean().item(),
        "normalized_gradient_mse": normalized_gradient_mse.item(),
    }
    return metrics, average_gradient


def main(cfg: EstimatorAuditConfig) -> None:
    if cfg.batch_size < 1 or cfg.num_batches < 1 or cfg.num_repeats < 2:
        raise ValueError("batch_size and num_batches must be positive; num_repeats must exceed 1")
    if cfg.num_cfm_samples < 2 or cfg.num_cfm_samples % 2 != 0:
        raise ValueError("num_cfm_samples must be an even integer of at least 2")

    torch.manual_seed(cfg.seed)
    metadata = LeRobotDatasetMetadata(cfg.dataset)
    policy = _load_policy(cfg, metadata)
    parameters = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
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

    batch_results = []
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= cfg.num_batches:
            break
        batch = _prepare_batch(raw_batch, cfg.device)
        iid_metrics, iid_average = _audit_mode(
            mode="iid",
            batch=batch,
            policy=policy,
            parameters=parameters,
            cfg=cfg,
            batch_index=batch_index,
        )
        antithetic_metrics, antithetic_average = _audit_mode(
            mode="joint_antithetic",
            batch=batch,
            policy=policy,
            parameters=parameters,
            cfg=cfg,
            batch_index=batch_index,
        )
        cross_mode_cosine = F.cosine_similarity(
            iid_average.unsqueeze(0), antithetic_average.unsqueeze(0), dim=1
        ).item()
        batch_results.append(
            {
                "batch": batch_index,
                "iid": iid_metrics,
                "joint_antithetic": antithetic_metrics,
                "cross_mode_average_gradient_cosine": cross_mode_cosine,
                "normalized_gradient_mse_relative_change": (
                    antithetic_metrics["normalized_gradient_mse"]
                    / iid_metrics["normalized_gradient_mse"]
                    - 1.0
                ),
            }
        )
        del iid_average, antithetic_average

    if len(batch_results) != cfg.num_batches:
        raise RuntimeError(f"expected {cfg.num_batches} batches, received {len(batch_results)}")
    result = {
        "checkpoint_path": str(Path(cfg.checkpoint_path).resolve()),
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "num_batches": cfg.num_batches,
        "num_cfm_samples": cfg.num_cfm_samples,
        "num_repeats": cfg.num_repeats,
        "num_parameters": sum(parameter.numel() for parameter in parameters),
        "batches": batch_results,
    }
    output_path = Path(cfg.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(EstimatorAuditConfig))
