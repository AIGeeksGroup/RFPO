import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from src.policy_averaging import average_state_dicts, create_averaged_checkpoint


def test_average_state_dicts_uniform_mean_and_metrics():
    states = [
        {
            "weight": torch.tensor([value, value + 2], dtype=torch.float16),
            "counter": torch.tensor([4], dtype=torch.int64),
        }
        for value in (0.0, 2.0, 4.0, 6.0)
    ]

    result, metrics = average_state_dicts(states)

    torch.testing.assert_close(
        result["weight"], torch.tensor([3.0, 5.0], dtype=torch.float16)
    )
    torch.testing.assert_close(result["counter"], states[0]["counter"])
    assert metrics["checkpoint_count"] == 4
    assert metrics["floating_tensor_count"] == 1
    assert metrics["copied_tensor_count"] == 1
    assert metrics["max_source_to_average_l2"] == pytest.approx(18**0.5)


def test_average_state_dicts_rejects_mismatch_and_nonfinite_values():
    with pytest.raises(ValueError, match="keys differ"):
        average_state_dicts([{"x": torch.zeros(1)}, {"y": torch.zeros(1)}])
    with pytest.raises(ValueError, match="non-floating tensor differs"):
        average_state_dicts(
            [
                {"counter": torch.tensor([1])},
                {"counter": torch.tensor([2])},
            ]
        )
    with pytest.raises(ValueError, match="non-finite"):
        average_state_dicts(
            [{"x": torch.zeros(1)}, {"x": torch.tensor([float("nan")])}]
        )


def test_average_state_dicts_rejects_shape_and_dtype_mismatches():
    with pytest.raises(ValueError, match="shape mismatch"):
        average_state_dicts([{"x": torch.zeros(1)}, {"x": torch.zeros(2)}])
    with pytest.raises(ValueError, match="dtype mismatch"):
        average_state_dicts(
            [{"x": torch.zeros(1)}, {"x": torch.zeros(1, dtype=torch.float64)}]
        )


def test_create_averaged_checkpoint(tmp_path):
    sources = []
    for index, value in enumerate((0.0, 2.0, 4.0, 6.0)):
        checkpoint = tmp_path / f"step_{index}"
        policy = checkpoint / "policy"
        policy.mkdir(parents=True)
        (policy / "config.json").write_text('{"type": "test"}\n', encoding="utf-8")
        save_file({"weight": torch.tensor([value])}, policy / "model.safetensors")
        sources.append(checkpoint)

    output = tmp_path / "swa"
    manifest = create_averaged_checkpoint(sources, output)

    state = load_file(output / "policy" / "model.safetensors")
    torch.testing.assert_close(state["weight"], torch.tensor([3.0]))
    assert json.loads((output / "policy" / "config.json").read_text()) == {
        "type": "test"
    }
    assert manifest["checkpoint_count"] == 4
    assert len(manifest["source_checkpoints"]) == 4
    with pytest.raises(FileExistsError):
        create_averaged_checkpoint(sources, output)


def test_create_averaged_checkpoint_rejects_config_mismatch(tmp_path):
    sources = []
    for index in range(2):
        checkpoint = tmp_path / f"step_{index}"
        policy = checkpoint / "policy"
        policy.mkdir(parents=True)
        (policy / "config.json").write_text(
            f'{{"index": {index}}}\n', encoding="utf-8"
        )
        save_file({"weight": torch.tensor([float(index)])}, policy / "model.safetensors")
        sources.append(checkpoint)

    with pytest.raises(ValueError, match="configs differ"):
        create_averaged_checkpoint(sources, tmp_path / "swa")
