# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    For FPO (flow-based) policies, this correctly handles the flow matching
    integration loop by using the policy's ``act_inference`` method and sizing
    the ONNX input to ``num_actor_obs`` (the actor observation dimension),
    rather than the raw MLP input dimension which also includes the timestep
    embedding and action dimensions.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file.

    For FPO policies, this stores the entire policy and delegates to
    ``act_inference`` so that the flow matching integration loop is
    included in the exported model.
    """

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        # Store the entire policy so act_inference (flow matching loop) is available
        self.policy = copy.deepcopy(policy)
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        return self.policy.act_inference(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file.

    For FPO policies, this stores the entire policy and delegates to
    ``act_inference`` so that the flow matching integration loop is
    included in the exported ONNX graph.  The dummy observation tensor
    used during tracing is sized to the actor's observation dimension
    rather than the raw MLP input dimension.
    """

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # Store the entire policy so act_inference (flow matching loop) is available
        self.policy = copy.deepcopy(policy)
        # Determine the true observation dimension for the ONNX input.
        # For recurrent policies, the observation feeds into the RNN first,
        # so the input dimension is the RNN input size.
        # For non-recurrent FPO policies, it is num_actor_obs.
        if self.is_recurrent and hasattr(policy, "memory_a"):
            self.num_obs = policy.memory_a.rnn.input_size
        else:
            self.num_obs = policy.num_actor_obs
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        return self.policy.act_inference(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        # Use the actor observation dimension (num_obs), NOT the raw MLP
        # input dimension (which for FPO includes timestep embedding + action dims).
        # This must match the normalizer's expected input shape.
        obs = torch.zeros(1, self.num_obs)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
