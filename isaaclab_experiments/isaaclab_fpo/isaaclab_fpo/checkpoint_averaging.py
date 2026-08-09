"""Strict actor-only checkpoint averaging for FPO locomotion policies."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


ACTOR_PREFIX = "actor."
EXPECTED_STEPS = (1300, 1350, 1400, 1450, 1499)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_exact(actual: Any, expected: Any, path: str = "root") -> None:
    """Recursively require equal types, keys, shapes, dtypes, and values."""
    if type(actual) is not type(expected):
        raise ValueError(
            f"{path} type mismatch: {type(actual).__name__} != "
            f"{type(expected).__name__}"
        )
    if isinstance(actual, torch.Tensor):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"{path} tensor metadata mismatch")
        if not torch.equal(actual, expected):
            raise ValueError(f"{path} tensor values changed")
        return
    if isinstance(actual, np.ndarray):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"{path} array metadata mismatch")
        if not np.array_equal(actual, expected, equal_nan=True):
            raise ValueError(f"{path} array values changed")
        return
    if isinstance(actual, Mapping):
        if list(actual.keys()) != list(expected.keys()):
            raise ValueError(f"{path} mapping keys or order changed")
        for key in actual:
            assert_exact(actual[key], expected[key], f"{path}[{key!r}]")
        return
    if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)):
        if len(actual) != len(expected):
            raise ValueError(f"{path} sequence length changed")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_exact(actual_item, expected_item, f"{path}[{index}]")
        return
    if isinstance(actual, float) and np.isnan(actual) and np.isnan(expected):
        return
    if actual != expected:
        raise ValueError(f"{path} value changed: {actual!r} != {expected!r}")


def _validate_model_layout(checkpoints: list[dict[str, Any]]) -> list[str]:
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if not all(isinstance(checkpoint, dict) for checkpoint in checkpoints):
        raise ValueError("every checkpoint must be a dictionary")
    if "model_state_dict" not in checkpoints[0]:
        raise ValueError("checkpoint is missing model_state_dict")

    reference_keys = list(checkpoints[0].keys())
    model = checkpoints[0]["model_state_dict"]
    if not isinstance(model, Mapping):
        raise ValueError("model_state_dict must be a mapping")
    model_keys = list(model.keys())
    actor_keys = [key for key in model_keys if key.startswith(ACTOR_PREFIX)]
    if not actor_keys:
        raise ValueError(f"model_state_dict has no {ACTOR_PREFIX!r} keys")

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        if list(checkpoint.keys()) != reference_keys:
            raise ValueError(f"checkpoint {checkpoint_index} top-level keys differ")
        candidate_model = checkpoint["model_state_dict"]
        if not isinstance(candidate_model, Mapping):
            raise ValueError(
                f"checkpoint {checkpoint_index} model_state_dict is not a mapping"
            )
        if list(candidate_model.keys()) != model_keys:
            raise ValueError(f"checkpoint {checkpoint_index} model keys differ")
        for key in model_keys:
            reference_tensor = model[key]
            candidate_tensor = candidate_model[key]
            if not isinstance(reference_tensor, torch.Tensor) or not isinstance(
                candidate_tensor, torch.Tensor
            ):
                raise ValueError(f"model_state_dict[{key!r}] must be a tensor")
            if (
                candidate_tensor.shape != reference_tensor.shape
                or candidate_tensor.dtype != reference_tensor.dtype
            ):
                raise ValueError(
                    f"checkpoint {checkpoint_index} tensor metadata differs for {key}"
                )
    for key in actor_keys:
        if not model[key].is_floating_point():
            raise ValueError(f"actor tensor {key} is not floating point")
    return actor_keys


def build_actor_average(
    checkpoints: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, torch.Tensor], float]:
    """Average actor tensors in float64 and preserve the final payload."""
    actor_keys = _validate_model_layout(checkpoints)
    final = checkpoints[-1]
    averaged: dict[str, torch.Tensor] = {}
    for key in actor_keys:
        tensors = [
            checkpoint["model_state_dict"][key].detach().cpu().to(torch.float64)
            for checkpoint in checkpoints
        ]
        mean = torch.stack(tensors, dim=0).mean(dim=0)
        if not torch.isfinite(mean).all():
            raise ValueError(f"averaged actor tensor {key} is non-finite")
        averaged[key] = mean.to(final["model_state_dict"][key].dtype)

    candidate = copy.deepcopy(final)
    for key, tensor in averaged.items():
        candidate["model_state_dict"][key] = tensor

    for key, value in final["model_state_dict"].items():
        if key not in averaged:
            assert_exact(candidate["model_state_dict"][key], value, f"model.{key}")
    for key, value in final.items():
        if key != "model_state_dict":
            assert_exact(candidate[key], value, key)

    final_flat = torch.cat(
        [final["model_state_dict"][key].detach().cpu().double().flatten() for key in actor_keys]
    )
    candidate_flat = torch.cat([averaged[key].double().flatten() for key in actor_keys])
    denominator = torch.linalg.vector_norm(final_flat)
    if denominator == 0:
        raise ValueError("final actor has zero L2 norm")
    relative_l2 = float(torch.linalg.vector_norm(candidate_flat - final_flat) / denominator)
    return candidate, averaged, relative_l2


def create_actor_average_checkpoint(
    source_paths: list[Path], output_path: Path, manifest_path: Path
) -> dict[str, Any]:
    """Create, reload, and validate an actor-average checkpoint without overwriting."""
    if len(source_paths) != 5:
        raise ValueError(f"H65 requires exactly five checkpoints, got {len(source_paths)}")
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("source checkpoint paths must be unique")
    for path in (output_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    expected_names = [f"model_{step}.pt" for step in EXPECTED_STEPS]
    actual_names = [path.name for path in source_paths]
    if actual_names != expected_names:
        raise ValueError(
            f"H65 source order must be {expected_names}, got {actual_names}"
        )
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False) for path in source_paths
    ]
    actual_steps = [checkpoint.get("iter") for checkpoint in checkpoints]
    if actual_steps != list(EXPECTED_STEPS):
        raise ValueError(
            f"H65 checkpoint iter fields must be {list(EXPECTED_STEPS)}, got {actual_steps}"
        )
    candidate, averaged, relative_l2 = build_actor_average(checkpoints)
    if not 0.01 <= relative_l2 <= 0.06:
        raise ValueError(
            f"candidate relative L2 {relative_l2:.8f} is outside [0.01, 0.06]"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as handle:
            torch.save(candidate, handle)
        reloaded = torch.load(output_path, map_location="cpu", weights_only=False)
        final = checkpoints[-1]
        actor_keys = list(averaged)
        for key in actor_keys:
            assert_exact(reloaded["model_state_dict"][key], averaged[key], f"model.{key}")
        for key, value in final["model_state_dict"].items():
            if key not in averaged:
                assert_exact(reloaded["model_state_dict"][key], value, f"model.{key}")
        for key, value in final.items():
            if key != "model_state_dict":
                assert_exact(reloaded[key], value, key)

        manifest = {
            "hypothesis": "H65",
            "algorithm": "uniform_float64_mean_actor_only",
            "sources": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in source_paths
            ],
            "final_container_source": str(source_paths[-1].resolve()),
            "output": str(output_path.resolve()),
            "output_sha256": _sha256(output_path),
            "actor_prefix": ACTOR_PREFIX,
            "actor_keys": actor_keys,
            "actor_tensor_count": len(actor_keys),
            "relative_l2_from_final": relative_l2,
            "relative_l2_gate": [0.01, 0.06],
            "non_actor_model_exact": True,
            "non_model_payload_exact": True,
            "passed": True,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    except Exception:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return manifest
