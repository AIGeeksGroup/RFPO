"""Source selection helpers for matched mirrored rollout evaluation."""

import torch


def select_rollout_source(
    mode: str,
    primary: torch.Tensor,
    secondary: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the exact source used by one independently executed trajectory."""
    if mode == "random":
        return primary
    if mode == "negative_random":
        return -primary
    if mode == "secondary_random":
        if secondary is None:
            raise ValueError("secondary_random requires a secondary source")
        return secondary
    raise ValueError(f"unsupported mirrored rollout mode: {mode}")
