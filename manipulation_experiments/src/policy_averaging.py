from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def average_state_dicts(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    if len(states) < 2:
        raise ValueError("checkpoint averaging requires at least two state dicts")

    reference = states[0]
    reference_keys = set(reference)
    for index, state in enumerate(states[1:], start=1):
        if set(state) != reference_keys:
            missing = sorted(reference_keys - set(state))
            extra = sorted(set(state) - reference_keys)
            raise ValueError(
                f"state-dict {index} keys differ: missing={missing}, extra={extra}"
            )

    averaged: dict[str, torch.Tensor] = {}
    floating_tensors = 0
    copied_tensors = 0
    for key in sorted(reference):
        tensors = [state[key] for state in states]
        reference_tensor = tensors[0]
        for index, tensor in enumerate(tensors[1:], start=1):
            if tensor.shape != reference_tensor.shape:
                raise ValueError(f"shape mismatch for {key} in state dict {index}")
            if tensor.dtype != reference_tensor.dtype:
                raise ValueError(f"dtype mismatch for {key} in state dict {index}")

        if reference_tensor.is_floating_point():
            if not all(torch.isfinite(tensor).all() for tensor in tensors):
                raise ValueError(f"non-finite floating tensor for {key}")
            value = tensors[0].float().clone()
            for tensor in tensors[1:]:
                value.add_(tensor.float())
            value.div_(len(tensors))
            if not torch.isfinite(value).all():
                raise ValueError(f"non-finite averaged tensor for {key}")
            averaged[key] = value.to(reference_tensor.dtype).contiguous()
            floating_tensors += 1
        else:
            if not all(torch.equal(reference_tensor, tensor) for tensor in tensors[1:]):
                raise ValueError(f"non-floating tensor differs for {key}")
            averaged[key] = reference_tensor.clone().contiguous()
            copied_tensors += 1

    squared_distances = [0.0 for _ in states]
    for key, mean_tensor in averaged.items():
        if not mean_tensor.is_floating_point():
            continue
        mean_float = mean_tensor.float()
        for index, state in enumerate(states):
            squared_distances[index] += (
                state[key].float() - mean_float
            ).square().double().sum().item()
    distances = [math.sqrt(value) for value in squared_distances]

    metrics: dict[str, float | int] = {
        "checkpoint_count": len(states),
        "floating_tensor_count": floating_tensors,
        "copied_tensor_count": copied_tensors,
        "mean_source_to_average_l2": sum(distances) / len(distances),
        "max_source_to_average_l2": max(distances),
    }
    return averaged, metrics


def _policy_dir(checkpoint: Path) -> Path:
    candidate = checkpoint / "policy"
    if candidate.is_dir():
        return candidate
    if (checkpoint / "model.safetensors").is_file():
        return checkpoint
    raise ValueError(f"policy checkpoint not found under {checkpoint}")


def create_averaged_checkpoint(
    source_checkpoints: Sequence[Path], output_checkpoint: Path
) -> dict[str, object]:
    if len(source_checkpoints) < 2:
        raise ValueError("checkpoint averaging requires at least two checkpoints")
    if output_checkpoint.exists():
        raise FileExistsError(f"output already exists: {output_checkpoint}")

    policy_dirs = [_policy_dir(path) for path in source_checkpoints]
    config_bytes = (policy_dirs[0] / "config.json").read_bytes()
    for policy_dir in policy_dirs[1:]:
        if (policy_dir / "config.json").read_bytes() != config_bytes:
            raise ValueError("source policy configs differ")

    states = [
        load_file(policy_dir / "model.safetensors", device="cpu")
        for policy_dir in policy_dirs
    ]
    averaged_state, metrics = average_state_dicts(states)

    output_policy = output_checkpoint / "policy"
    output_policy.mkdir(parents=True)
    shutil.copy2(policy_dirs[0] / "config.json", output_policy / "config.json")
    save_file(
        averaged_state,
        output_policy / "model.safetensors",
        metadata={"format": "pt", "method": "uniform_checkpoint_average"},
    )

    manifest: dict[str, object] = {
        "method": "uniform_policy_checkpoint_average",
        "source_checkpoints": [str(path.resolve()) for path in source_checkpoints],
        **metrics,
    }
    (output_checkpoint / "averaging_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
