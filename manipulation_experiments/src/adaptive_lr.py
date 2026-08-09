from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveLearningRateDecision:
    learning_rate: float
    action: str


def adapt_learning_rate_from_kl(
    learning_rate: float,
    kl_proxy: float,
    *,
    desired_kl: float,
    factor: float = 1.5,
    minimum: float,
    maximum: float,
) -> AdaptiveLearningRateDecision:
    """Apply the released FPO++ two-sided KL learning-rate rule."""
    if learning_rate <= 0 or desired_kl <= 0 or factor <= 1:
        raise ValueError("learning rate and desired KL must be positive; factor must exceed one")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid learning-rate bounds")
    if kl_proxy < 0:
        raise ValueError("KL proxy must be non-negative")

    if kl_proxy > desired_kl * 2.0:
        return AdaptiveLearningRateDecision(
            max(minimum, learning_rate / factor), "decrease"
        )
    if 0.0 < kl_proxy < desired_kl / 2.0:
        return AdaptiveLearningRateDecision(
            min(maximum, learning_rate * factor), "increase"
        )
    return AdaptiveLearningRateDecision(learning_rate, "hold")

