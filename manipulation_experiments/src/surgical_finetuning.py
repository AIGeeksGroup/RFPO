"""Parameter-scope selection and movement audits for surgical actor fine-tuning."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


ActorTrainableScope = Literal["all", "output_head"]


def configure_actor_trainable_scope(
    actor: nn.Module, scope: ActorTrainableScope
) -> dict[str, object]:
    """Apply an actor parameter scope and return an auditable manifest."""
    if scope not in ("all", "output_head"):
        raise ValueError(f"unsupported actor trainable scope: {scope}")

    if scope == "output_head":
        model = getattr(actor, "model", None)
        config = getattr(actor, "config", None)
        architecture = getattr(config, "network_architecture", None)
        mlp = getattr(model, "mlp", None)
        if architecture != "mlp" or not isinstance(mlp, nn.Sequential):
            raise ValueError("output_head scope requires an MLP flow actor")
        if not len(mlp) or not isinstance(mlp[-1], nn.Linear):
            raise ValueError("output_head scope requires the final MLP module to be nn.Linear")

        for parameter in actor.parameters():
            parameter.requires_grad = False
        for parameter in mlp[-1].parameters():
            parameter.requires_grad = True

    named = list(actor.named_parameters())
    trainable_names = [name for name, parameter in named if parameter.requires_grad]
    frozen_names = [name for name, parameter in named if not parameter.requires_grad]
    trainable_count = sum(parameter.numel() for _, parameter in named if parameter.requires_grad)
    total_count = sum(parameter.numel() for _, parameter in named)
    if not trainable_names:
        raise ValueError(f"actor trainable scope {scope} selected no parameters")
    if scope == "output_head" and not all(
        name.startswith(f"model.mlp.{len(actor.model.mlp) - 1}.")
        for name in trainable_names
    ):
        raise RuntimeError("output_head scope selected parameters outside the final MLP layer")
    return {
        "scope": scope,
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "trainable_fraction": trainable_count / total_count,
    }


def snapshot_actor_parameters(actor: nn.Module) -> dict[str, torch.Tensor]:
    """Copy actor parameters to CPU for an exact post-update movement audit."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in actor.named_parameters()
    }


def actor_parameter_movement(
    actor: nn.Module, initial: dict[str, torch.Tensor]
) -> dict[str, object]:
    """Summarize maximum movement separately for trainable and frozen parameters."""
    current = dict(actor.named_parameters())
    if current.keys() != initial.keys():
        raise ValueError("actor parameter names changed after the initial snapshot")

    trainable_max = 0.0
    frozen_max = 0.0
    moved_trainable_names: list[str] = []
    moved_frozen_names: list[str] = []
    for name, parameter in current.items():
        displacement = float(
            (parameter.detach().cpu() - initial[name]).abs().max().item()
        )
        if parameter.requires_grad:
            trainable_max = max(trainable_max, displacement)
            if displacement > 0:
                moved_trainable_names.append(name)
        else:
            frozen_max = max(frozen_max, displacement)
            if displacement > 0:
                moved_frozen_names.append(name)
    return {
        "max_abs_trainable_displacement": trainable_max,
        "max_abs_frozen_displacement": frozen_max,
        "moved_trainable_names": moved_trainable_names,
        "moved_frozen_names": moved_frozen_names,
        "frozen_parameters_bitwise_unchanged": not moved_frozen_names,
    }


def optimizer_parameter_names(
    actor: nn.Module, optimizer: torch.optim.Optimizer
) -> list[str]:
    """Resolve optimizer parameter identities back to stable actor parameter names."""
    names_by_id = {id(parameter): name for name, parameter in actor.named_parameters()}
    names: list[str] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in names_by_id:
                raise ValueError("actor optimizer contains a parameter outside the actor")
            names.append(names_by_id[id(parameter)])
    if len(names) != len(set(names)):
        raise ValueError("actor optimizer contains duplicate parameters")
    return names
