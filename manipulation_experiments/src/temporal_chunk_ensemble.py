"""ACT-inspired averaging for two overlapping action chunks."""

from dataclasses import dataclass
from math import exp

import torch
from torch import Tensor


ACT_TEMPORAL_DECAY = 0.01
ACT_OLD_WEIGHT = 1.0 / (1.0 + exp(-ACT_TEMPORAL_DECAY))
ACT_NEW_WEIGHT = 1.0 - ACT_OLD_WEIGHT


@dataclass(frozen=True)
class TemporalChunkEnsembleResult:
    executed: Tensor
    continuation: Tensor
    blended: bool


def ensemble_overlapping_chunks(
    current_prediction: Tensor,
    previous_continuation: Tensor | None,
    *,
    action_steps: int,
) -> TemporalChunkEnsembleResult:
    """Blend one old continuation with the overlapping prefix of a new prediction."""
    if current_prediction.ndim != 2:
        raise ValueError("current_prediction must have shape (horizon, action_dim)")
    if action_steps < 1 or current_prediction.shape[0] < 2 * action_steps:
        raise ValueError("prediction horizon must contain two complete action chunks")

    current_prefix = current_prediction[:action_steps]
    next_continuation = current_prediction[action_steps : 2 * action_steps].clone()
    if previous_continuation is None:
        return TemporalChunkEnsembleResult(
            executed=current_prefix.clone(),
            continuation=next_continuation,
            blended=False,
        )
    if previous_continuation.shape != current_prefix.shape:
        raise ValueError(
            "previous_continuation must match the current action prefix shape"
        )
    if previous_continuation.device != current_prefix.device:
        raise ValueError("overlapping chunks must be on the same device")
    if previous_continuation.dtype != current_prefix.dtype:
        raise ValueError("overlapping chunks must have the same dtype")

    executed = ACT_OLD_WEIGHT * previous_continuation + ACT_NEW_WEIGHT * current_prefix
    return TemporalChunkEnsembleResult(
        executed=executed,
        continuation=next_continuation,
        blended=True,
    )
