import pytest

from src.heldout_ratio_early_stop import select_epoch_before_ratio_violation


def test_selects_snapshot_before_first_violation():
    assert select_epoch_before_ratio_violation([0.95, 0.84, 0.79, 0.82], 0.80) == 2


def test_selects_final_epoch_when_all_are_safe():
    assert select_epoch_before_ratio_violation([0.95, 0.80, 0.81], 0.80) == 3


def test_selects_preupdate_when_first_epoch_violates():
    assert select_epoch_before_ratio_violation([0.79, 0.75], 0.80) == 0


@pytest.mark.parametrize(
    ("fractions", "threshold"),
    [([], 0.8), ([0.9], 0.0), ([1.1], 0.8)],
)
def test_rejects_invalid_inputs(fractions, threshold):
    with pytest.raises(ValueError):
        select_epoch_before_ratio_violation(fractions, threshold)
