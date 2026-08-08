import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from src.policy_interpolation import (
    create_interpolated_checkpoint,
    interpolate_state_dicts,
)


def test_interpolate_state_dicts_midpoint_and_metrics():
    anchor = {
        "weight": torch.tensor([1.0, 3.0]),
        "counter": torch.tensor([4], dtype=torch.int64),
    }
    finetuned = {
        "weight": torch.tensor([5.0, 7.0]),
        "counter": torch.tensor([4], dtype=torch.int64),
    }

    result, metrics = interpolate_state_dicts(anchor, finetuned, alpha=0.5)

    torch.testing.assert_close(result["weight"], torch.tensor([3.0, 5.0]))
    torch.testing.assert_close(result["counter"], anchor["counter"])
    assert metrics["floating_tensor_count"] == 1
    assert metrics["copied_tensor_count"] == 1
    assert metrics["relative_anchor_distance"] == pytest.approx(0.5)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_interpolate_state_dicts_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        interpolate_state_dicts({"x": torch.zeros(1)}, {"x": torch.ones(1)}, alpha)


def test_interpolate_state_dicts_rejects_changed_nonfloating_tensor():
    with pytest.raises(ValueError, match="non-floating tensor differs"):
        interpolate_state_dicts(
            {"counter": torch.tensor([1])},
            {"counter": torch.tensor([2])},
            alpha=0.5,
        )


def test_create_interpolated_checkpoint(tmp_path):
    anchor = tmp_path / "anchor"
    finetuned = tmp_path / "finetuned"
    output = tmp_path / "midpoint"
    for checkpoint, value in ((anchor, 0.0), (finetuned, 2.0)):
        policy = checkpoint / "policy"
        policy.mkdir(parents=True)
        (policy / "config.json").write_text('{"type": "test"}\n', encoding="utf-8")
        save_file({"weight": torch.tensor([value])}, policy / "model.safetensors")

    manifest = create_interpolated_checkpoint(anchor, finetuned, output, alpha=0.5)

    state = load_file(output / "policy" / "model.safetensors")
    torch.testing.assert_close(state["weight"], torch.tensor([1.0]))
    assert json.loads((output / "policy" / "config.json").read_text()) == {
        "type": "test"
    }
    assert manifest["relative_anchor_distance"] == pytest.approx(0.5)
    with pytest.raises(FileExistsError):
        create_interpolated_checkpoint(anchor, finetuned, output, alpha=0.5)
