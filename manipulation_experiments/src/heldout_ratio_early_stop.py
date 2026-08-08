"""Selection helpers for held-out CFM ratio early stopping."""

from __future__ import annotations

from collections.abc import Sequence


def select_epoch_before_ratio_violation(
    active_fractions: Sequence[float],
    threshold: float,
) -> int:
    """Return the last safe epoch before the first threshold violation.

    Epoch zero denotes the pre-update snapshot. If no completed epoch violates the
    threshold, the final completed epoch is returned.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if not active_fractions:
        raise ValueError("at least one completed epoch is required")
    for epoch, fraction in enumerate(active_fractions, start=1):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("active fractions must lie in [0, 1]")
        if fraction < threshold:
            return epoch - 1
    return len(active_fractions)
