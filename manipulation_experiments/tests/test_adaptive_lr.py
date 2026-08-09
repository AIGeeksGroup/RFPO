import pytest

from src.adaptive_lr import adapt_learning_rate_from_kl


def test_adaptive_lr_matches_released_threshold_rule():
    kwargs = dict(desired_kl=1e-4, minimum=1e-6, maximum=1e-3)

    assert adapt_learning_rate_from_kl(1e-5, 3e-4, **kwargs).action == "decrease"
    assert adapt_learning_rate_from_kl(1e-5, 3e-4, **kwargs).learning_rate == pytest.approx(
        1e-5 / 1.5
    )
    assert adapt_learning_rate_from_kl(1e-5, 2e-5, **kwargs).action == "increase"
    assert adapt_learning_rate_from_kl(1e-5, 2e-5, **kwargs).learning_rate == pytest.approx(
        1.5e-5
    )
    assert adapt_learning_rate_from_kl(1e-5, 1e-4, **kwargs).action == "hold"
    assert adapt_learning_rate_from_kl(1e-5, 0.0, **kwargs).action == "hold"


def test_adaptive_lr_respects_relative_transfer_bounds():
    kwargs = dict(desired_kl=1e-4, minimum=1e-6, maximum=1e-3)

    low = adapt_learning_rate_from_kl(1e-6, 3e-4, **kwargs)
    high = adapt_learning_rate_from_kl(1e-3, 2e-5, **kwargs)

    assert low.learning_rate == 1e-6
    assert high.learning_rate == 1e-3

