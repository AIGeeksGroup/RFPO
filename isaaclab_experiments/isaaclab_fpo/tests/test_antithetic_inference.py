import pytest
import torch

from isaaclab_fpo.antithetic_inference import (
    antithetic_action,
    antithetic_action_batched,
    paired_source_action,
)


class QuadraticPolicy:
    def __init__(self):
        self.sources = []

    def act_inference(self, observations, *, source, **_kwargs):
        self.sources.append(source.clone())
        return observations[:, : source.shape[1]] + source + 0.25 * source.square()


def test_antithetic_action_uses_exact_pair_and_float_mean():
    policy = QuadraticPolicy()
    observations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    source = torch.tensor([[0.5, -1.0], [2.0, -0.25]])

    result = antithetic_action(policy, observations, source)

    assert len(policy.sources) == 2
    assert torch.equal(policy.sources[0], source)
    assert torch.equal(policy.sources[1], -source)
    positive = observations + source + 0.25 * source.square()
    negative = observations - source + 0.25 * source.square()
    assert torch.equal(result, (positive + negative) * 0.5)


def test_antithetic_action_rejects_nonfinite_source():
    with pytest.raises(ValueError, match="finite"):
        antithetic_action(
            QuadraticPolicy(), torch.zeros(1, 2), torch.tensor([[0.0, float("nan")]])
        )


def test_batched_antithetic_action_uses_one_doubled_call():
    policy = QuadraticPolicy()
    observations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    source = torch.tensor([[0.5, -1.0], [2.0, -0.25]])

    result = antithetic_action_batched(policy, observations, source)

    assert len(policy.sources) == 1
    assert torch.equal(policy.sources[0], torch.cat((source, -source), dim=0))
    positive = observations + source + 0.25 * source.square()
    negative = observations - source + 0.25 * source.square()
    assert torch.equal(result, (positive + negative) * 0.5)


def test_paired_source_action_averages_independent_endpoints():
    policy = QuadraticPolicy()
    observations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    first = torch.tensor([[0.5, -1.0], [2.0, -0.25]])
    second = torch.tensor([[-0.2, 0.4], [0.3, 1.5]])

    result = paired_source_action(policy, observations, first, second)

    assert torch.equal(policy.sources[0], first)
    assert torch.equal(policy.sources[1], second)
    first_endpoint = observations + first + 0.25 * first.square()
    second_endpoint = observations + second + 0.25 * second.square()
    assert torch.equal(result, (first_endpoint + second_endpoint) * 0.5)


def test_paired_source_action_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        paired_source_action(
            QuadraticPolicy(),
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.zeros(2, 3),
        )
