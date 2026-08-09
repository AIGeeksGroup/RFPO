import pytest
import torch
from torch import nn

from src.surgical_finetuning import (
    actor_parameter_movement,
    configure_actor_trainable_scope,
    optimizer_parameter_names,
    snapshot_actor_parameters,
)


class TinyActor(nn.Module):
    def __init__(self, architecture="mlp"):
        super().__init__()
        self.config = type("Config", (), {"network_architecture": architecture})()
        self.model = nn.Module()
        self.model.vision_encoder = nn.Linear(3, 3)
        self.model.mlp = nn.Sequential(
            nn.Linear(4, 5), nn.Mish(), nn.Linear(5, 2)
        )


def test_output_head_scope_selects_only_final_linear_layer():
    actor = TinyActor()
    manifest = configure_actor_trainable_scope(actor, "output_head")

    assert manifest["trainable_names"] == ["model.mlp.2.weight", "model.mlp.2.bias"]
    assert manifest["trainable_parameter_count"] == 12
    assert all(
        parameter.requires_grad == name.startswith("model.mlp.2.")
        for name, parameter in actor.named_parameters()
    )


def test_output_head_receives_gradient_and_frozen_parameters_do_not_move():
    actor = TinyActor()
    configure_actor_trainable_scope(actor, "output_head")
    initial = snapshot_actor_parameters(actor)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=1e-2,
    )

    actor.model.mlp(torch.randn(8, 4)).square().mean().backward()
    assert actor.model.mlp[-1].weight.grad is not None
    optimizer.step()

    movement = actor_parameter_movement(actor, initial)
    assert movement["max_abs_trainable_displacement"] > 0
    assert movement["max_abs_frozen_displacement"] == 0
    assert movement["frozen_parameters_bitwise_unchanged"]
    assert optimizer_parameter_names(actor, optimizer) == [
        "model.mlp.2.weight",
        "model.mlp.2.bias",
    ]


@pytest.mark.parametrize("architecture", ["unet", "residual_mlp"])
def test_output_head_rejects_non_mlp_architectures(architecture):
    with pytest.raises(ValueError, match="requires an MLP flow actor"):
        configure_actor_trainable_scope(TinyActor(architecture), "output_head")


def test_output_head_rejects_missing_final_linear_layer():
    actor = TinyActor()
    actor.model.mlp = nn.Sequential(nn.Linear(4, 5), nn.Mish())
    with pytest.raises(ValueError, match="final MLP module"):
        configure_actor_trainable_scope(actor, "output_head")


def test_all_scope_preserves_existing_requires_grad_selection():
    actor = TinyActor()
    actor.model.vision_encoder.weight.requires_grad = False
    manifest = configure_actor_trainable_scope(actor, "all")
    assert "model.vision_encoder.weight" in manifest["frozen_names"]
    assert "model.mlp.2.weight" in manifest["trainable_names"]
