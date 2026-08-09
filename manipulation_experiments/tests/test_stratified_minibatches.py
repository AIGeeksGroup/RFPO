import numpy as np
import pytest

from src.stratified_minibatches import advantage_sign_stratified_permutation


def test_stratified_permutation_balances_signs_and_covers_samples():
    advantages = np.concatenate((np.arange(13) + 1.0, -np.arange(27) - 1.0))

    permutation, metrics = advantage_sign_stratified_permutation(
        advantages, num_minibatches=8, rng=np.random.default_rng(7)
    )

    assert np.array_equal(np.sort(permutation), np.arange(40))
    batches = permutation.reshape(8, 5)
    positive_counts = (advantages[batches] > 0).sum(axis=1)
    assert positive_counts.min() >= 1
    assert positive_counts.max() - positive_counts.min() <= 1
    assert np.array_equal(
        positive_counts, np.asarray(metrics["positive_counts_by_minibatch"])
    )
    assert metrics["unique_index_count"] == 40


@pytest.mark.parametrize(
    "advantages,batches,message",
    [
        (np.ones(8), 1, "at least two"),
        (np.ones(9), 2, "divisible"),
        (np.array([1.0, -1.0, float("nan"), -2.0]), 2, "finite"),
        (np.array([1.0] * 3 + [-1.0] * 13), 4, "support every"),
    ],
)
def test_stratified_permutation_rejects_invalid_inputs(
    advantages, batches, message
):
    with pytest.raises(ValueError, match=message):
        advantage_sign_stratified_permutation(
            advantages, num_minibatches=batches, rng=np.random.default_rng(0)
        )
