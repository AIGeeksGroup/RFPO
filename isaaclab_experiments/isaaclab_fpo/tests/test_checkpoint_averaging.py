import copy

import pytest
import torch

from isaaclab_fpo.checkpoint_averaging import (
    assert_exact,
    build_actor_average,
    create_actor_average_checkpoint,
)


def _checkpoint(value: float):
    return {
        "model_state_dict": {
            "actor.0.weight": torch.tensor([[value, value + 1]], dtype=torch.float32),
            "critic.0.weight": torch.tensor([[9.0, 10.0]], dtype=torch.float32),
        },
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(7)}}},
        "iter": 1499,
        "infos": {"tag": "final"},
        "obs_norm_state_dict": {"mean": torch.tensor([3.0])},
    }


def test_build_actor_average_changes_only_actor():
    checkpoints = [_checkpoint(float(index)) for index in range(5)]
    final_before = copy.deepcopy(checkpoints[-1])

    candidate, averaged, relative_l2 = build_actor_average(checkpoints)

    assert torch.equal(averaged["actor.0.weight"], torch.tensor([[2.0, 3.0]]))
    assert torch.equal(
        candidate["model_state_dict"]["actor.0.weight"],
        averaged["actor.0.weight"],
    )
    assert_exact(candidate["model_state_dict"]["critic.0.weight"], final_before["model_state_dict"]["critic.0.weight"])
    for key in final_before:
        if key != "model_state_dict":
            assert_exact(candidate[key], final_before[key])
    assert relative_l2 > 0
    assert_exact(checkpoints[-1], final_before)


def test_layout_mismatch_is_rejected():
    checkpoints = [_checkpoint(float(index)) for index in range(5)]
    checkpoints[2]["model_state_dict"]["actor.0.weight"] = torch.ones(3)

    with pytest.raises(ValueError, match="tensor metadata differs"):
        build_actor_average(checkpoints)


def test_creator_refuses_existing_output_before_loading(tmp_path):
    output = tmp_path / "candidate.pt"
    manifest = tmp_path / "manifest.json"
    output.write_bytes(b"occupied")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_actor_average_checkpoint(
            [tmp_path / f"source_{index}.pt" for index in range(5)], output, manifest
        )


def test_creator_round_trips_candidate_and_manifest(tmp_path):
    sources = []
    for step, value in zip((1300, 1350, 1400, 1450, 1499), (3.6, 3.7, 3.8, 3.9, 4.0)):
        path = tmp_path / f"model_{step}.pt"
        checkpoint = _checkpoint(value)
        checkpoint["iter"] = step
        torch.save(checkpoint, path)
        sources.append(path)
    output = tmp_path / "candidate.pt"
    manifest_path = tmp_path / "manifest.json"

    manifest = create_actor_average_checkpoint(sources, output, manifest_path)
    candidate = torch.load(output, map_location="cpu", weights_only=False)

    assert manifest["passed"] is True
    assert 0.01 <= manifest["relative_l2_from_final"] <= 0.06
    assert output.is_file()
    assert manifest_path.is_file()
    assert torch.equal(
        candidate["model_state_dict"]["actor.0.weight"],
        torch.tensor([[3.8, 4.8]], dtype=torch.float32),
    )
