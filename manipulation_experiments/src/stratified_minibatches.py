from __future__ import annotations

from typing import Protocol

import numpy as np


class _NumpyShuffle(Protocol):
    def shuffle(self, values: np.ndarray) -> None: ...


def advantage_sign_stratified_permutation(
    advantages: np.ndarray,
    num_minibatches: int,
    rng: _NumpyShuffle,
) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(advantages).reshape(-1)
    sample_count = values.size
    if num_minibatches < 2 or sample_count == 0:
        raise ValueError("requires samples and at least two minibatches")
    if sample_count % num_minibatches:
        raise ValueError("sample count must be divisible by num_minibatches")
    if not np.isfinite(values).all():
        raise ValueError("advantages must be finite")

    minibatch_size = sample_count // num_minibatches
    positive = np.flatnonzero(values > 0)
    non_positive = np.flatnonzero(values <= 0)
    if min(positive.size, non_positive.size) < num_minibatches:
        raise ValueError("each advantage sign must support every minibatch")

    rng.shuffle(positive)
    rng.shuffle(non_positive)
    positive_counts = np.full(
        num_minibatches, positive.size // num_minibatches, dtype=np.int64
    )
    positive_counts[: positive.size % num_minibatches] += 1
    rng.shuffle(positive_counts)
    if np.any(positive_counts <= 0) or np.any(positive_counts >= minibatch_size):
        raise ValueError("both advantage signs must occur in every minibatch")

    batches: list[np.ndarray] = []
    positive_start = 0
    non_positive_start = 0
    for positive_count in positive_counts:
        non_positive_count = minibatch_size - int(positive_count)
        batch = np.concatenate(
            (
                positive[positive_start : positive_start + positive_count],
                non_positive[
                    non_positive_start : non_positive_start + non_positive_count
                ],
            )
        )
        rng.shuffle(batch)
        batches.append(batch)
        positive_start += int(positive_count)
        non_positive_start += non_positive_count

    permutation = np.concatenate(batches)
    if not np.array_equal(np.sort(permutation), np.arange(sample_count)):
        raise RuntimeError("stratified permutation did not cover each sample exactly once")

    metrics: dict[str, object] = {
        "sample_count": sample_count,
        "minibatch_count": num_minibatches,
        "minibatch_size": minibatch_size,
        "positive_count": int(positive.size),
        "non_positive_count": int(non_positive.size),
        "positive_counts_by_minibatch": positive_counts.tolist(),
        "positive_count_range": int(positive_counts.max() - positive_counts.min()),
        "unique_index_count": int(np.unique(permutation).size),
    }
    return permutation, metrics
