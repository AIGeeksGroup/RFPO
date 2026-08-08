import pytest
import torch

from src.flow_metrics import flow_geometry


def test_flow_geometry_is_exact_for_a_straight_path():
    source = torch.zeros(2, 1, 1)
    endpoint = torch.tensor([[[2.0]], [[-4.0]]])
    alpha = torch.tensor([0.25, 0.5, 0.75, 1.0]).view(1, 4, 1, 1)
    path = source[:, None] + alpha * (endpoint - source)[:, None]

    metrics = flow_geometry(source, path)

    torch.testing.assert_close(metrics["straightness_error"], torch.zeros(2))
    torch.testing.assert_close(metrics["path_length_ratio"], torch.ones(2))


def test_flow_geometry_detects_a_bent_path():
    source = torch.zeros(1, 1, 2)
    path = torch.tensor([[[[0.5, 1.0]], [[1.0, 0.0]]]])

    metrics = flow_geometry(source, path)

    assert metrics["straightness_error"].item() == pytest.approx(0.5)
    assert metrics["path_length_ratio"].item() > 1.0


def test_flow_geometry_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="dimensions must match"):
        flow_geometry(torch.zeros(2, 3, 4), torch.zeros(1, 5, 3, 4))
