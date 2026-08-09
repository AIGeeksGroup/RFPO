import pytest
import torch

from src.temporal_chunk_ensemble import (
    ACT_NEW_WEIGHT,
    ACT_OLD_WEIGHT,
    ensemble_overlapping_chunks,
)


def test_first_chunk_is_bitwise_unmodified_and_does_not_alias_prediction():
    prediction = torch.arange(32, dtype=torch.float32).reshape(16, 2)

    result = ensemble_overlapping_chunks(
        prediction, None, action_steps=8
    )

    assert not result.blended
    assert torch.equal(result.executed, prediction[:8])
    assert torch.equal(result.continuation, prediction[8:])
    assert result.executed.data_ptr() != prediction.data_ptr()
    result.executed.zero_()
    assert torch.equal(prediction[:8], torch.arange(16).reshape(8, 2))


def test_second_chunk_uses_locked_act_weights_and_saves_only_new_tail():
    prediction = torch.arange(32, dtype=torch.float64).reshape(16, 2)
    old = torch.full((8, 2), -4.0, dtype=torch.float64)

    result = ensemble_overlapping_chunks(
        prediction, old, action_steps=8
    )

    expected = ACT_OLD_WEIGHT * old + ACT_NEW_WEIGHT * prediction[:8]
    assert result.blended
    torch.testing.assert_close(result.executed, expected)
    assert torch.equal(result.continuation, prediction[8:])
    assert ACT_OLD_WEIGHT == pytest.approx(0.5024999791668749)
    assert ACT_NEW_WEIGHT == pytest.approx(0.4975000208331251)


@pytest.mark.parametrize(
    ("prediction", "continuation", "message"),
    [
        (torch.zeros(16), None, "current_prediction"),
        (torch.zeros(15, 2), None, "two complete"),
        (torch.zeros(16, 2), torch.zeros(7, 2), "must match"),
    ],
)
def test_invalid_chunk_shapes_are_rejected(prediction, continuation, message):
    with pytest.raises(ValueError, match=message):
        ensemble_overlapping_chunks(prediction, continuation, action_steps=8)
