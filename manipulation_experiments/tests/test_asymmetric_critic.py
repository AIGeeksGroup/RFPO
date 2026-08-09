import pytest
import torch

from src.asymmetric_critic import (
    discounted_success_labels,
    normalize_from_train,
    stratified_episode_folds,
    validate_collection,
)


def test_discounted_success_labels_respect_macro_durations():
    labels = discounted_success_labels([2, 3], [0.0, 0.9**2], discount=0.9)
    torch.testing.assert_close(labels, torch.tensor([0.9**4, 0.9**2]))


def test_stratified_episode_folds_are_disjoint_and_balanced():
    outcomes = {seed: seed >= 8 for seed in range(16)}
    first, second = stratified_episode_folds(outcomes)
    assert first | second == set(outcomes)
    assert not first & second
    assert sum(outcomes[seed] for seed in first) == 4
    assert sum(outcomes[seed] for seed in second) == 4


def test_normalization_uses_training_statistics_only():
    train = torch.tensor([[0.0, 2.0], [2.0, 4.0]])
    evaluation = torch.tensor([[4.0, 8.0]])
    normalized_train, normalized_evaluation = normalize_from_train(train, evaluation)
    torch.testing.assert_close(normalized_train.mean(0), torch.zeros(2))
    torch.testing.assert_close(normalized_evaluation, torch.tensor([[3.0, 5.0]]))


def test_collection_validation_requires_outcome_support():
    records = []
    for seed in range(16):
        records.append(
            {
                "seed": seed,
                "visual": torch.ones(2),
                "privileged": torch.ones(3),
                "terminal": True,
                "success": seed < 8,
            }
        )
    outcomes = validate_collection(records, set(range(16)))
    assert sum(outcomes.values()) == 8

    records[-1]["success"] = True
    with pytest.raises(RuntimeError, match="outcome support"):
        validate_collection(records, set(range(16)))
