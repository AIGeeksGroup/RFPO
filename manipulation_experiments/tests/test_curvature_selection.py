import pytest
import torch

from src.curvature_selection import select_lower_curvature_candidate


def _linear_path(source: torch.Tensor, endpoint: torch.Tensor, steps: int) -> torch.Tensor:
    alpha = torch.linspace(1 / steps, 1, steps).view(1, steps, 1, 1)
    return source[:, None] + alpha * (endpoint - source)[:, None]


def test_selects_complete_lower_curvature_candidate_per_batch_item():
    sources = torch.zeros(2, 2, 3, 1)
    endpoints = torch.tensor(
        [[[[1.0], [1.0], [1.0]], [[2.0], [2.0], [2.0]]]]
    ).expand(2, -1, -1, -1).clone()
    paths = _linear_path(sources.flatten(0, 1), endpoints.flatten(0, 1), 2).reshape(
        2, 2, 2, 3, 1
    )
    paths[0, 1, 0, 0, 0] += 1.0
    paths[1, 0, 0, 0, 0] += 1.0
    actions = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3, 1)

    selected_actions, selected_paths, metrics = select_lower_curvature_candidate(
        sources, paths, actions, action_steps=3
    )

    assert metrics["selected_indices"].tolist() == [0, 1]
    assert torch.equal(selected_actions[0], actions[0, 0])
    assert torch.equal(selected_actions[1], actions[1, 1])
    assert torch.equal(selected_paths[0], paths[0, 0])
    assert torch.equal(selected_paths[1], paths[1, 1])
    assert torch.equal(metrics["selected_scores"], metrics["scores"].min(dim=1).values)


def test_scores_only_the_executed_action_prefix():
    sources = torch.zeros(1, 2, 3, 1)
    endpoints = torch.ones_like(sources)
    paths = _linear_path(sources.flatten(0, 1), endpoints.flatten(0, 1), 2).reshape(
        1, 2, 2, 3, 1
    )
    paths[0, 0, 0, 2, 0] += 2.0
    actions = torch.zeros_like(sources)

    _, _, prefix_metrics = select_lower_curvature_candidate(
        sources, paths, actions, action_steps=2
    )
    _, _, full_metrics = select_lower_curvature_candidate(
        sources, paths, actions, action_steps=3
    )

    assert prefix_metrics["selected_indices"].item() == 0
    assert full_metrics["selected_indices"].item() == 1


@pytest.mark.parametrize("candidate_count", [1, 3])
def test_rejects_candidate_counts_other_than_two(candidate_count):
    sources = torch.zeros(1, candidate_count, 2, 1)
    paths = torch.zeros(1, candidate_count, 2, 2, 1)
    actions = torch.zeros_like(sources)
    with pytest.raises(ValueError, match="exactly two"):
        select_lower_curvature_candidate(
            sources, paths, actions, action_steps=2
        )
