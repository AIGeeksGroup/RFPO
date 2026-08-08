from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file


def interpolate_state_dicts(
    anchor: Mapping[str, torch.Tensor],
    finetuned: Mapping[str, torch.Tensor],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if set(anchor) != set(finetuned):
        missing = sorted(set(anchor) - set(finetuned))
        extra = sorted(set(finetuned) - set(anchor))
        raise ValueError(f"state-dict keys differ: missing={missing}, extra={extra}")

    result: dict[str, torch.Tensor] = {}
    squared_endpoint_distance = 0.0
    squared_anchor_distance = 0.0
    floating_tensors = 0
    copied_tensors = 0

    for key in sorted(anchor):
        anchor_tensor = anchor[key]
        finetuned_tensor = finetuned[key]
        if anchor_tensor.shape != finetuned_tensor.shape:
            raise ValueError(
                f"shape mismatch for {key}: {anchor_tensor.shape} != {finetuned_tensor.shape}"
            )
        if anchor_tensor.dtype != finetuned_tensor.dtype:
            raise ValueError(
                f"dtype mismatch for {key}: {anchor_tensor.dtype} != {finetuned_tensor.dtype}"
            )

        if anchor_tensor.is_floating_point():
            anchor_float = anchor_tensor.float()
            delta = finetuned_tensor.float() - anchor_float
            interpolated = anchor_float + alpha * delta
            result[key] = interpolated.to(anchor_tensor.dtype).contiguous()
            squared_endpoint_distance += delta.square().double().sum().item()
            squared_anchor_distance += (
                result[key].float() - anchor_float
            ).square().double().sum().item()
            floating_tensors += 1
        else:
            if not torch.equal(anchor_tensor, finetuned_tensor):
                raise ValueError(f"non-floating tensor differs for {key}")
            result[key] = anchor_tensor.clone().contiguous()
            copied_tensors += 1

    endpoint_distance = math.sqrt(squared_endpoint_distance)
    anchor_distance = math.sqrt(squared_anchor_distance)
    metrics: dict[str, float | int] = {
        "floating_tensor_count": floating_tensors,
        "copied_tensor_count": copied_tensors,
        "anchor_to_finetuned_l2": endpoint_distance,
        "anchor_to_interpolated_l2": anchor_distance,
        "relative_anchor_distance": (
            anchor_distance / endpoint_distance if endpoint_distance > 0.0 else 0.0
        ),
    }
    return result, metrics


def _policy_dir(checkpoint: Path) -> Path:
    candidate = checkpoint / "policy"
    if candidate.is_dir():
        return candidate
    if (checkpoint / "model.safetensors").is_file():
        return checkpoint
    raise ValueError(f"policy checkpoint not found under {checkpoint}")


def create_interpolated_checkpoint(
    anchor_checkpoint: Path,
    finetuned_checkpoint: Path,
    output_checkpoint: Path,
    alpha: float,
) -> dict[str, object]:
    anchor_policy = _policy_dir(anchor_checkpoint)
    finetuned_policy = _policy_dir(finetuned_checkpoint)
    if output_checkpoint.exists():
        raise FileExistsError(f"output already exists: {output_checkpoint}")

    anchor_config = anchor_policy / "config.json"
    finetuned_config = finetuned_policy / "config.json"
    if anchor_config.read_bytes() != finetuned_config.read_bytes():
        raise ValueError("anchor and finetuned policy configs differ")

    anchor_state = load_file(anchor_policy / "model.safetensors", device="cpu")
    finetuned_state = load_file(finetuned_policy / "model.safetensors", device="cpu")
    interpolated_state, metrics = interpolate_state_dicts(
        anchor_state, finetuned_state, alpha
    )

    output_policy = output_checkpoint / "policy"
    output_policy.mkdir(parents=True)
    shutil.copy2(anchor_config, output_policy / "config.json")
    save_file(
        interpolated_state,
        output_policy / "model.safetensors",
        metadata={"format": "pt", "interpolation_alpha": str(alpha)},
    )

    manifest: dict[str, object] = {
        "method": "linear_policy_weight_interpolation",
        "alpha_finetuned": alpha,
        "anchor_checkpoint": str(anchor_checkpoint.resolve()),
        "finetuned_checkpoint": str(finetuned_checkpoint.resolve()),
        **metrics,
    }
    (output_checkpoint / "interpolation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
