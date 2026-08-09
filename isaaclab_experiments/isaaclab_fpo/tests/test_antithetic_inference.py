import pytest
import torch

from isaaclab_fpo.antithetic_inference import antithetic_action


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
        antithetic_action(QuadraticPolicy(), torch.zeros(1, 2), torch.tensor([[0.0, float("nan")]]))
