import pytest
import torch

from src.antithetic_inference import average_antithetic_predictions, build_antithetic_sources


def test_build_antithetic_sources_is_exact_and_does_not_alias_input():
    source = torch.tensor([[[-2.0, 0.5], [1.0, 3.0]]])

    positive, negative = build_antithetic_sources(source)

    torch.testing.assert_close(positive, source)
    torch.testing.assert_close(negative, -source)
    assert positive.data_ptr() != source.data_ptr()
    positive.zero_()
    torch.testing.assert_close(source, torch.tensor([[[-2.0, 0.5], [1.0, 3.0]]]))


def test_average_antithetic_predictions_is_elementwise_mean():
    positive = torch.tensor([[[2.0, 4.0], [-2.0, 8.0]]])
    negative = torch.tensor([[[0.0, 8.0], [6.0, 2.0]]])

    result = average_antithetic_predictions(positive, negative)

    torch.testing.assert_close(result, torch.tensor([[[1.0, 6.0], [2.0, 5.0]]]))


def test_average_antithetic_predictions_rejects_broadcasting():
    with pytest.raises(ValueError, match="identical shapes"):
        average_antithetic_predictions(torch.zeros(2, 3), torch.zeros(1, 3))
