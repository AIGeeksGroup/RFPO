import torch

from src.bc_anchor_pcgrad import (
    gradient_cosine,
    gradient_norm,
    project_conflicting_gradient,
)


def test_projection_removes_conflicting_anchor_component():
    primary = [torch.tensor([1.0, -2.0])]
    anchor = [torch.tensor([0.0, 3.0])]

    projected, dot, conflict = project_conflicting_gradient(primary, anchor)

    assert conflict
    torch.testing.assert_close(dot, torch.tensor(-6.0))
    torch.testing.assert_close(projected[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(
        gradient_cosine(projected, anchor), torch.tensor(0.0), atol=1e-7, rtol=0.0
    )


def test_projection_leaves_nonconflicting_gradient_unchanged():
    primary = [torch.tensor([1.0, 2.0]), None]
    anchor = [torch.tensor([0.0, 3.0]), torch.tensor([4.0])]

    projected, dot, conflict = project_conflicting_gradient(primary, anchor)

    assert not conflict
    torch.testing.assert_close(dot, torch.tensor(6.0))
    torch.testing.assert_close(projected[0], primary[0])
    assert projected[1] is None
    torch.testing.assert_close(gradient_norm(projected), torch.sqrt(torch.tensor(5.0)))


def test_projection_handles_missing_anchor_gradient():
    primary = [torch.tensor([1.0]), torch.tensor([-2.0])]
    anchor = [None, torch.tensor([1.0])]

    projected, _, conflict = project_conflicting_gradient(primary, anchor)

    assert conflict
    torch.testing.assert_close(projected[0], primary[0])
    torch.testing.assert_close(projected[1], torch.tensor([0.0]))


def test_projected_negative_branch_no_longer_cancels_positive_branch():
    positive = torch.tensor([2.0, 0.0])
    negative = torch.tensor([-1.0, 3.0])

    projected, dot, conflict = project_conflicting_gradient(
        [negative], [positive]
    )
    candidate = positive + projected[0]

    assert conflict
    torch.testing.assert_close(dot, torch.tensor(-2.0))
    torch.testing.assert_close(projected[0], torch.tensor([0.0, 3.0]))
    torch.testing.assert_close(candidate, torch.tensor([2.0, 3.0]))
    assert torch.dot(candidate, positive) > 0
