from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor


BASE_NOISE_SEED = 20261062


def keyed_seed(*parts: int, base_seed: int = BASE_NOISE_SEED) -> int:
    """Map an integer experiment key to a stable positive torch seed."""
    digest = hashlib.sha256()
    digest.update(int(base_seed).to_bytes(8, "little", signed=False))
    for part in parts:
        digest.update(int(part).to_bytes(8, "little", signed=True))
    return int.from_bytes(digest.digest()[:8], "little") % (2**63 - 1)


def keyed_standard_normal(
    shape: tuple[int, ...],
    *parts: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("shape must contain positive dimensions")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(keyed_seed(*parts))
    return torch.randn(shape, generator=generator, dtype=dtype, device="cpu")


def array_sha256(value: np.ndarray | Tensor) -> str:
    if isinstance(value, Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def candidate_rms(first: np.ndarray, second: np.ndarray) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape or first_array.size == 0:
        raise ValueError("candidate chunks must have the same nonempty shape")
    return float(np.sqrt(np.mean(np.square(second_array - first_array))))


def _validate_differences(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size < 4:
        raise ValueError(f"{name} must contain at least four values")
    if not np.isfinite(array).all() or np.any(np.abs(array) > 1.0):
        raise ValueError(f"{name} must contain finite values in [-1, 1]")
    return array


def summarize_vine_differences(
    block_a: Iterable[float],
    block_b: Iterable[float],
) -> dict[str, object]:
    differences_a = _validate_differences(block_a, "block_a")
    differences_b = _validate_differences(block_b, "block_b")
    if differences_a.shape != differences_b.shape:
        raise ValueError("continuation blocks must cover the same vine states")

    nonzero_a = int(np.count_nonzero(differences_a))
    nonzero_b = int(np.count_nonzero(differences_b))
    common = (differences_a != 0) & (differences_b != 0)
    common_count = int(common.sum())
    sign_agreement = (
        float(np.mean(np.sign(differences_a[common]) == np.sign(differences_b[common])))
        if common_count
        else 0.0
    )
    correlation = (
        float(np.corrcoef(differences_a, differences_b)[0, 1])
        if np.std(differences_a) > 0 and np.std(differences_b) > 0
        else float("nan")
    )
    top_count = max(1, math.ceil(differences_a.size / 4))
    top_a = np.argsort(-differences_a, kind="stable")[:top_count]
    top_b = np.argsort(-differences_b, kind="stable")[:top_count]
    heldout_b = float(differences_b[top_a].mean())
    heldout_a = float(differences_a[top_b].mean())

    gates = {
        "nonzero_block_a": nonzero_a >= 8,
        "nonzero_block_b": nonzero_b >= 8,
        "common_nonzero_count": common_count >= 6,
        "pearson_correlation": np.isfinite(correlation) and correlation >= 0.35,
        "common_sign_agreement": common_count >= 6 and sign_agreement >= 0.75,
        "top_a_heldout_b": heldout_b > 0,
        "top_b_heldout_a": heldout_a > 0,
    }
    return {
        "differences_block_a": differences_a.tolist(),
        "differences_block_b": differences_b.tolist(),
        "nonzero_block_a": nonzero_a,
        "nonzero_block_b": nonzero_b,
        "common_nonzero_count": common_count,
        "pearson_correlation": correlation,
        "common_sign_agreement": sign_agreement,
        "top_quartile_count": top_count,
        "top_a_indices": top_a.tolist(),
        "top_b_indices": top_b.tolist(),
        "top_a_heldout_b": heldout_b,
        "top_b_heldout_a": heldout_a,
        "gates": gates,
        "passed": all(gates.values()),
    }


def summarize_vine_records(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    for record in records:
        key = (int(record["root_seed"]), int(record["branch_step"]))
        outcome_key = (int(record["candidate"]), int(record["continuation_replica"]))
        success = int(record["success"])
        if success not in (0, 1):
            raise ValueError("vine outcomes must be binary")
        outcomes = grouped.setdefault(key, {})
        if outcome_key in outcomes:
            raise ValueError(f"duplicate vine outcome for {key} and {outcome_key}")
        outcomes[outcome_key] = success

    expected = {(candidate, replica) for candidate in (0, 1) for replica in range(4)}
    state_keys = sorted(grouped)
    if not state_keys:
        raise ValueError("at least one vine state is required")
    differences_a = []
    differences_b = []
    state_rows = []
    for key in state_keys:
        outcomes = grouped[key]
        if set(outcomes) != expected:
            raise ValueError(f"incomplete vine outcomes for {key}")
        candidate_0_a = np.mean([outcomes[(0, replica)] for replica in (0, 1)])
        candidate_1_a = np.mean([outcomes[(1, replica)] for replica in (0, 1)])
        candidate_0_b = np.mean([outcomes[(0, replica)] for replica in (2, 3)])
        candidate_1_b = np.mean([outcomes[(1, replica)] for replica in (2, 3)])
        difference_a = float(candidate_1_a - candidate_0_a)
        difference_b = float(candidate_1_b - candidate_0_b)
        differences_a.append(difference_a)
        differences_b.append(difference_b)
        state_rows.append(
            {
                "root_seed": key[0],
                "branch_step": key[1],
                "candidate_0_block_a": float(candidate_0_a),
                "candidate_1_block_a": float(candidate_1_a),
                "candidate_0_block_b": float(candidate_0_b),
                "candidate_1_block_b": float(candidate_1_b),
                "difference_block_a": difference_a,
                "difference_block_b": difference_b,
            }
        )
    return {
        "num_vine_states": len(state_keys),
        "states": state_rows,
        **summarize_vine_differences(differences_a, differences_b),
    }
