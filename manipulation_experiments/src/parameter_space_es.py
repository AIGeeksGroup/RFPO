from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor, nn


PERTURBED_PARAMETER_NAMES = (
    "model.mlp.0.weight",
    "model.mlp.2.weight",
    "model.mlp.4.weight",
    "model.mlp.6.weight",
)


def snapshot_parameters(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def validate_parameter_family(
    module: nn.Module,
    names: tuple[str, ...] = PERTURBED_PARAMETER_NAMES,
) -> dict[str, nn.Parameter]:
    parameters = dict(module.named_parameters())
    missing = sorted(set(names) - set(parameters))
    if missing:
        raise ValueError(f"missing locked ES parameters: {missing}")
    selected = {name: parameters[name] for name in names}
    non_matrices = [name for name, parameter in selected.items() if parameter.ndim != 2]
    if non_matrices:
        raise ValueError(f"locked ES parameters must be matrices: {non_matrices}")
    return selected


def gaussian_direction(
    anchor: Mapping[str, Tensor],
    seed: int,
    relative_scale: float = 0.01,
    names: tuple[str, ...] = PERTURBED_PARAMETER_NAMES,
) -> dict[str, Tensor]:
    if relative_scale <= 0:
        raise ValueError("relative_scale must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction: dict[str, Tensor] = {}
    for name in names:
        if name not in anchor:
            raise ValueError(f"anchor is missing locked parameter {name}")
        parameter = anchor[name]
        noise = torch.randn(
            parameter.shape,
            generator=generator,
            dtype=torch.float64,
            device="cpu",
        )
        noise_norm = torch.linalg.vector_norm(noise)
        parameter_norm = torch.linalg.vector_norm(parameter.detach().double()).cpu()
        if noise_norm == 0 or parameter_norm == 0:
            raise ValueError(f"cannot normalize zero-norm tensor {name}")
        scaled = noise * (relative_scale * parameter_norm / noise_norm)
        direction[name] = scaled.to(device=parameter.device, dtype=parameter.dtype)
    return direction


@torch.no_grad()
def restore_anchor(module: nn.Module, anchor: Mapping[str, Tensor]) -> None:
    parameters = dict(module.named_parameters())
    if set(parameters) != set(anchor):
        raise ValueError("module parameters do not match the anchor manifest")
    for name, parameter in parameters.items():
        parameter.copy_(anchor[name])


@torch.no_grad()
def apply_mirrored_direction(
    module: nn.Module,
    anchor: Mapping[str, Tensor],
    direction: Mapping[str, Tensor],
    sign: int,
) -> None:
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    restore_anchor(module, anchor)
    parameters = dict(module.named_parameters())
    if set(direction) != set(PERTURBED_PARAMETER_NAMES):
        raise ValueError("direction does not match the locked parameter family")
    for name, delta in direction.items():
        parameters[name].add_(delta, alpha=sign)


def displacement_manifest(
    module: nn.Module,
    anchor: Mapping[str, Tensor],
) -> dict[str, object]:
    parameters = dict(module.named_parameters())
    relative_norms: dict[str, float] = {}
    anchor_norms: dict[str, float] = {}
    displacement_norms: dict[str, float] = {}
    unlisted_unchanged = True
    finite = True
    for name, parameter in parameters.items():
        finite = finite and bool(torch.isfinite(parameter).all().item())
        if name in PERTURBED_PARAMETER_NAMES:
            displacement = (parameter.detach() - anchor[name]).double()
            anchor_norm = torch.linalg.vector_norm(anchor[name].double())
            displacement_norm = torch.linalg.vector_norm(displacement)
            anchor_norms[name] = float(anchor_norm.item())
            displacement_norms[name] = float(displacement_norm.item())
            relative_norms[name] = float((displacement_norm / anchor_norm).item())
        else:
            unlisted_unchanged = unlisted_unchanged and torch.equal(
                parameter.detach(), anchor[name]
            )
    return {
        "finite": finite,
        "anchor_norms": anchor_norms,
        "displacement_norms": displacement_norms,
        "relative_norms": relative_norms,
        "unlisted_parameters_bitwise_unchanged": unlisted_unchanged,
    }


def anchor_restored_exactly(module: nn.Module, anchor: Mapping[str, Tensor]) -> bool:
    return all(
        torch.equal(parameter.detach(), anchor[name])
        for name, parameter in module.named_parameters()
    )


def tensor_sha256(tensor: Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def parameter_hashes(
    tensors: Mapping[str, Tensor],
    names: tuple[str, ...] | None = None,
) -> dict[str, str]:
    selected_names = tuple(tensors) if names is None else names
    return {name: tensor_sha256(tensors[name]) for name in selected_names}


def batched_observation_hashes(observation: Mapping[str, Tensor]) -> list[str]:
    if not observation:
        raise ValueError("observation must not be empty")
    batch_sizes = {int(value.shape[0]) for value in observation.values()}
    if len(batch_sizes) != 1:
        raise ValueError("all observation tensors must share a batch dimension")
    batch_size = batch_sizes.pop()
    hashes = []
    for batch_index in range(batch_size):
        digest = hashlib.sha256()
        for key in sorted(observation):
            value = observation[key][batch_index].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        hashes.append(digest.hexdigest())
    return hashes


def half_paired_rms(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.shape != negative.shape or positive.size == 0:
        raise ValueError("paired action arrays must have the same nonempty shape")
    return float(np.sqrt(np.mean(np.square(0.5 * (positive - negative)))))


def evaluate_signal(successes: np.ndarray) -> dict[str, object]:
    """Evaluate locked H53 gates for [direction, sign(-,+), eight seeds]."""
    values = np.asarray(successes, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 8):
        raise ValueError("successes must have shape [directions, 2, 8]")
    if values.shape[0] < 4:
        raise ValueError("at least four directions are required for top-quartile gates")
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("successes must be finite binary values")

    differences = values[:, 1] - values[:, 0]
    d_a = differences[:, :4].mean(axis=1)
    d_b = differences[:, 4:].mean(axis=1)
    nonzero_a = int(np.count_nonzero(d_a))
    nonzero_b = int(np.count_nonzero(d_b))
    common = (d_a != 0) & (d_b != 0)
    common_count = int(common.sum())
    sign_agreement = (
        float(np.mean(np.sign(d_a[common]) == np.sign(d_b[common])))
        if common_count
        else 0.0
    )
    if np.std(d_a) == 0 or np.std(d_b) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(d_a, d_b)[0, 1])
    top_a = np.argsort(-d_a, kind="stable")[:4]
    top_b = np.argsort(-d_b, kind="stable")[:4]
    heldout_b = float(d_b[top_a].mean())
    heldout_a = float(d_a[top_b].mean())

    gates = {
        "nonzero_replica_a": nonzero_a >= 6,
        "nonzero_replica_b": nonzero_b >= 6,
        "replica_correlation": np.isfinite(correlation) and correlation >= 0.35,
        "common_nonzero_count": common_count >= 5,
        "common_sign_agreement": common_count >= 5 and sign_agreement >= 0.70,
        "top_a_heldout_b": heldout_b >= 0.125,
        "top_b_heldout_a": heldout_a >= 0.125,
    }
    return {
        "d_a": d_a.tolist(),
        "d_b": d_b.tolist(),
        "nonzero_a": nonzero_a,
        "nonzero_b": nonzero_b,
        "correlation": correlation,
        "common_nonzero_count": common_count,
        "common_sign_agreement": sign_agreement,
        "top_a_indices": top_a.tolist(),
        "top_b_indices": top_b.tolist(),
        "top_a_heldout_b_mean": heldout_b,
        "top_b_heldout_a_mean": heldout_a,
        "gates": gates,
        "all_signal_gates_pass": all(gates.values()),
    }
