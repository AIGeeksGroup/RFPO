from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algos.rl.fpo_core import (
    FPOPolicyConfig,
    FPOStatePolicy,
    chunked_fpo_surrogate_loss,
    fpo_surrogate_loss,
    gaussian_ppo_surrogate_loss,
    sample_cfm_tensors,
)
from algos.rl.fpo_rollout_buffer import FPOTransition, FPORolloutBuffer


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _params_to_dict(params: Any) -> dict[str, Any]:
    def to_plain(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [to_plain(v) for v in value]
        if hasattr(value, "items"):
            return {str(k): to_plain(v) for k, v in value.items()}
        return value

    if isinstance(params, dict):
        return {str(k): to_plain(v) for k, v in params.items()}
    if hasattr(params, "items"):
        return {str(k): to_plain(v) for k, v in params.items()}
    return {
        key: to_plain(getattr(params, key))
        for key in dir(params)
        if not key.startswith("_") and not callable(getattr(params, key))
    }


_FORBIDDEN_PHASE_KEYS = {
    "bc_checkpoint",
    "bc_anchor_dataset",
    "ppo_base_checkpoint",
    "ppo_teacher_checkpoint",
    "resume_model",
}


def _normalise_phase_schedule(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        if raw.strip() in ("", "[]", "null", "None"):
            return []
        raise ValueError("agent.params.phase_schedule must be a list, not a string")

    phases: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        phase = _params_to_dict(item)
        meta_keys = {"name", "start_step", "start", "values", "reset_optimizer"}
        values = phase.get("values", None)
        if values is None:
            values = {key: value for key, value in phase.items() if key not in meta_keys}
        else:
            values = _params_to_dict(values)
        forbidden = sorted((set(phase) | set(values)) & _FORBIDDEN_PHASE_KEYS)
        if forbidden:
            raise ValueError(f"phase_schedule cannot set warm-start fields: {', '.join(forbidden)}")
        start_step = int(phase.get("start_step", phase.get("start", 0)))
        if start_step < 0:
            raise ValueError("phase_schedule start_step must be non-negative")
        phases.append(
            {
                "name": str(phase.get("name", f"phase_{index}")),
                "start_step": start_step,
                "reset_optimizer": bool(phase.get("reset_optimizer", False)),
                "values": values,
            }
        )

    return sorted(phases, key=lambda phase: phase["start_step"])


def _phase_values_for_timestep(schedule: list[dict[str, Any]], num_timesteps: int) -> dict[str, Any]:
    active_values: dict[str, Any] = {}
    for phase in schedule:
        if int(num_timesteps) < int(phase["start_step"]):
            break
        active_values = dict(phase.get("values", {}))
    return active_values


def _policy_config_from_params(params: Any, obs_dim: int, action_dim: int) -> FPOPolicyConfig:
    return FPOPolicyConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_hidden_dims=tuple(getattr(params, "actor_hidden_dims", (256, 128))),
        critic_hidden_dims=tuple(getattr(params, "critic_hidden_dims", (256, 128))),
        activation=str(getattr(params, "activation", "tanh")),
        timestep_embed_dim=int(getattr(params, "timestep_embed_dim", 16)),
        sampling_steps=int(getattr(params, "sampling_steps", 8)),
        actor_mlp_output_scale=float(getattr(params, "actor_mlp_output_scale", 1.0)),
        actor_scale=float(getattr(params, "actor_scale", 1.0)),
        action_clip=float(getattr(params, "action_clip", 1.0)),
        action_perturb_std=float(getattr(params, "action_perturb_std", 0.0)),
        cfm_loss_t_inverse_cdf_beta=float(getattr(params, "cfm_loss_t_inverse_cdf_beta", 1.5)),
        cfm_loss_reduction=str(getattr(params, "cfm_loss_reduction", "mean")),
        cfm_loss_use_huber=bool(getattr(params, "cfm_loss_use_huber", True)),
        cfm_loss_huber_delta=float(getattr(params, "cfm_loss_huber_delta", 1.0)),
        cfm_loss_huber_style=str(getattr(params, "cfm_loss_huber_style", "torch")),
        flow_network_output_param=str(getattr(params, "flow_network_output_param", "u")),
        cfm_loss_mode=str(getattr(params, "cfm_loss_mode", "u")),
        action_head_mode=str(getattr(params, "action_head_mode", "flow_residual")),
        residual_head_hidden_dims=tuple(getattr(params, "residual_head_hidden_dims", (128, 64))),
        residual_action_scale=float(getattr(params, "residual_action_scale", 0.1)),
        residual_head_zero_init=bool(getattr(params, "residual_head_zero_init", True)),
        direct_head_hidden_dims=tuple(getattr(params, "direct_head_hidden_dims", (256, 128))),
        direct_action_scale=float(getattr(params, "direct_action_scale", 1.0)),
        direct_head_zero_init=bool(getattr(params, "direct_head_zero_init", False)),
    )


def _merge_checkpoint_policy_config(
    checkpoint_config: dict[str, Any],
    params: Any,
    obs_dim: int,
    action_dim: int,
) -> FPOPolicyConfig:
    """Keep BC checkpoint architecture but allow runtime flow-loss/sampling overrides."""
    clean = dict(checkpoint_config)
    runtime_config = _policy_config_from_params(params, obs_dim, action_dim).to_dict()
    override_keys = (
        "sampling_steps",
        "actor_mlp_output_scale",
        "actor_scale",
        "action_clip",
        "action_perturb_std",
        "cfm_loss_t_inverse_cdf_beta",
        "cfm_loss_reduction",
        "cfm_loss_use_huber",
        "cfm_loss_huber_delta",
        "cfm_loss_huber_style",
        "flow_network_output_param",
        "cfm_loss_mode",
        "action_head_mode",
        "residual_head_hidden_dims",
        "residual_action_scale",
        "residual_head_zero_init",
        "direct_head_hidden_dims",
        "direct_action_scale",
        "direct_head_zero_init",
    )
    for key in override_keys:
        clean[key] = runtime_config[key]
    return FPOPolicyConfig.from_mapping(clean)


def _policy_configs_have_same_architecture(left: FPOPolicyConfig, right: FPOPolicyConfig) -> bool:
    architecture_keys = (
        "obs_dim",
        "action_dim",
        "actor_hidden_dims",
        "critic_hidden_dims",
        "activation",
        "timestep_embed_dim",
        "residual_head_hidden_dims",
    )
    return all(getattr(left, key) == getattr(right, key) for key in architecture_keys)


def should_advance_vividex_curriculum(
    *,
    current_stage: int,
    eval_metrics: dict[str, Any],
    pregrasp_threshold: float = 0.95,
    stage1_metric: str | None = None,
    stage1_threshold: float = 0.0,
    stage2_metric: str | None = None,
    stage2_threshold: float = 0.0,
) -> bool:
    if current_stage > 1:
        return False
    pregrasp = eval_metrics.get("eval/mean_pregrasp_success")
    if pregrasp is None or float(pregrasp) <= pregrasp_threshold:
        return False
    if current_stage == 0 and stage1_metric:
        metric_value = eval_metrics.get(stage1_metric)
        return metric_value is not None and float(metric_value) > float(stage1_threshold)
    if current_stage == 1 and stage2_metric:
        metric_value = eval_metrics.get(stage2_metric)
        return metric_value is not None and float(metric_value) > float(stage2_threshold)
    return True


def linear_bc_anchor_coef(
    initial_coef: float,
    decay_steps: int,
    num_timesteps: int,
    min_coef: float = 0.0,
) -> float:
    initial_coef = float(initial_coef)
    decay_steps = int(decay_steps)
    min_coef = max(0.0, min(float(min_coef), initial_coef))
    if initial_coef <= 0:
        return 0.0
    if decay_steps <= 0:
        return initial_coef
    progress = min(max(float(num_timesteps) / float(decay_steps), 0.0), 1.0)
    return min_coef + (initial_coef - min_coef) * (1.0 - progress)


def _linear_coef_from_params(
    *,
    initial_coef: float,
    decay_steps: int,
    num_timesteps: int,
    min_coef: float = 0.0,
) -> float:
    return linear_bc_anchor_coef(
        initial_coef=initial_coef,
        decay_steps=decay_steps,
        num_timesteps=num_timesteps,
        min_coef=min_coef,
    )


def _scaled_action_std(
    action_std: float | torch.Tensor,
    *,
    scale: float,
    min_scale: float,
    decay_steps: int,
    num_timesteps: int,
) -> float | torch.Tensor:
    scale_now = linear_bc_anchor_coef(
        initial_coef=float(scale),
        decay_steps=int(decay_steps),
        num_timesteps=int(num_timesteps),
        min_coef=float(min_scale),
    )
    if isinstance(action_std, torch.Tensor):
        return action_std * scale_now
    return float(action_std) * scale_now


def _initial_log_std(value: Any, action_dim: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, str):
        if value.lower() == "ppo_base":
            raise RuntimeError("gaussian_action_std=ppo_base is only valid with gaussian_action_std_source=ppo_base")
        value = float(value)
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(action_dim)
    tensor = tensor.reshape(-1)
    if tensor.numel() != action_dim:
        raise ValueError(f"Expected gaussian_action_std with {action_dim} values, got {tensor.numel()}")
    return tensor.clamp_min(1e-6).log()


@dataclass
class LoadedFPOPolicy:
    policy: FPOStatePolicy
    num_timesteps: int
    extra: dict[str, Any]
    gaussian_log_std: torch.Tensor | None = None

    def _action_std(self) -> float | torch.Tensor:
        agent_params = self.extra.get("agent_params", {})
        if not isinstance(agent_params, dict):
            return 0.05
        source = str(agent_params.get("gaussian_action_std_source", "fixed")).lower()
        gaussian_action_std = agent_params.get("gaussian_action_std", 0.05)
        if source == "ppo_base" or str(gaussian_action_std).lower() == "ppo_base":
            base_policy = getattr(self.policy, "base_policy", None)
            log_std = getattr(getattr(base_policy, "policy", None), "log_std", None)
            if log_std is None:
                raise RuntimeError("gaussian_action_std_source=ppo_base requires a PPO base policy with log_std")
            action_std = torch.exp(log_std.detach()).to(device=next(self.policy.parameters()).device)
            return _scaled_action_std(
                action_std,
                scale=float(agent_params.get("gaussian_action_std_scale", 1.0)),
                min_scale=float(agent_params.get("gaussian_action_std_min_scale", 0.0)),
                decay_steps=int(agent_params.get("gaussian_action_std_decay_steps", 0)),
                num_timesteps=self.num_timesteps,
            )
        if source != "fixed":
            raise ValueError(f"Unknown gaussian_action_std_source: {source}")
        if self.gaussian_log_std is not None:
            action_std = self.gaussian_log_std.exp().to(device=next(self.policy.parameters()).device)
            return _scaled_action_std(
                action_std,
                scale=float(agent_params.get("gaussian_action_std_scale", 1.0)),
                min_scale=float(agent_params.get("gaussian_action_std_min_scale", 0.0)),
                decay_steps=int(agent_params.get("gaussian_action_std_decay_steps", 0)),
                num_timesteps=self.num_timesteps,
            )
        return _scaled_action_std(
            float(gaussian_action_std),
            scale=float(agent_params.get("gaussian_action_std_scale", 1.0)),
            min_scale=float(agent_params.get("gaussian_action_std_min_scale", 0.0)),
            decay_steps=int(agent_params.get("gaussian_action_std_decay_steps", 0)),
            num_timesteps=self.num_timesteps,
        )

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = True):
        single_obs = False
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
            single_obs = True
        obs_tensor = torch.as_tensor(obs, device=next(self.policy.parameters()).device)
        agent_params = self.extra.get("agent_params", {})
        actor_objective = ""
        if isinstance(agent_params, dict):
            actor_objective = str(agent_params.get("actor_objective", "")).lower()
        was_training = self.policy.training
        self.policy.eval()
        with torch.no_grad():
            if not deterministic and actor_objective == "gaussian_ppo":
                actions, _, _ = self.policy.sample_gaussian_action(obs_tensor, action_std=self._action_std())
            else:
                actions = self.policy.act(obs_tensor, deterministic=deterministic)
        if was_training:
            self.policy.train()
        actions_np = actions.detach().cpu().numpy()
        if single_obs:
            actions_np = actions_np[0]
        return actions_np, state


def save_fpo_checkpoint(
    *,
    path: str,
    policy: FPOStatePolicy,
    optimizer_state_dict: dict[str, Any] | None,
    num_timesteps: int,
    extra: dict[str, Any] | None = None,
    gaussian_log_std: torch.Tensor | None = None,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "algorithm_name": "FPO",
            "policy_config": policy.cfg.to_dict(),
            "policy_state_dict": policy.state_dict(),
            "base_policy_state_dict": (
                policy.base_policy.policy.state_dict()
                if policy.base_policy_trainable
                and policy.base_policy is not None
                and hasattr(policy.base_policy, "policy")
                else None
            ),
            "optimizer_state_dict": optimizer_state_dict,
            "gaussian_log_std": None if gaussian_log_std is None else gaussian_log_std.detach().cpu(),
            "normalizers": policy.normalizer_state_dict(),
            "num_timesteps": int(num_timesteps),
            "extra": dict(extra or {}),
        },
        path,
    )


def load_fpo_state_policy(path: str, device: str | torch.device = "auto") -> LoadedFPOPolicy:
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    checkpoint = torch.load(path, map_location=device)
    cfg = FPOPolicyConfig.from_mapping(checkpoint["policy_config"])
    policy = FPOStatePolicy(cfg).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    normalizers = checkpoint.get("normalizers")
    if normalizers:
        policy.set_normalizers(
            normalizers["obs_mean"],
            normalizers["obs_std"],
            normalizers["action_mean"],
            normalizers["action_std"],
        )
    extra = dict(checkpoint.get("extra", {}))
    agent_params = extra.get("agent_params", {}) if isinstance(extra.get("agent_params", {}), dict) else {}
    base_checkpoint = agent_params.get("ppo_base_checkpoint")
    if base_checkpoint:
        from stable_baselines3 import PPO

        base_policy = PPO.load(base_checkpoint, device=device)
        trainable_base = bool(agent_params.get("trainable_ppo_base_actor", False))
        for param in base_policy.policy.parameters():
            param.requires_grad = False
        policy.set_residual_base(
            base_policy,
            residual_coef=float(agent_params.get("flow_residual_coef", 1.0)),
            trainable=trainable_base,
        )
        base_state = checkpoint.get("base_policy_state_dict")
        if base_state is not None:
            base_policy.policy.load_state_dict(base_state, strict=True)
    gaussian_log_std = checkpoint.get("gaussian_log_std")
    if gaussian_log_std is not None:
        gaussian_log_std = torch.as_tensor(gaussian_log_std, dtype=torch.float32, device=device)
    policy.eval()
    return LoadedFPOPolicy(
        policy=policy,
        num_timesteps=int(checkpoint.get("num_timesteps", 0)),
        extra=extra,
        gaussian_log_std=gaussian_log_std,
    )


class FPOStateTrainer:
    def __init__(self, env, cfg, output_dir: str, device: str = "auto"):
        self.env = env
        self.cfg = cfg
        self.output_dir = output_dir
        self.device = self._resolve_device(device)
        self.logger = None

        params = cfg.agent.params
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        if obs_shape is None or action_shape is None:
            raise ValueError("FPO state trainer requires dense Box spaces")
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(np.prod(action_shape))
        self.n_envs = int(cfg.n_envs)
        self.n_steps = max(1, int(params.n_steps // cfg.n_envs))
        self.batch_size = max(1, int(params.batch_size))
        collected = self.n_envs * self.n_steps
        self.num_minibatches = max(1, int(np.ceil(collected / self.batch_size)))
        self.num_epochs = int(params.n_epochs)
        self.gamma = float(params.gamma)
        self.gae_lambda = float(params.gae_lambda)
        self.learning_rate = float(params.learning_rate)
        self.vf_coef = float(params.vf_coef)
        self.clip_range = float(params.clip_range)
        self.max_grad_norm = float(params.max_grad_norm)
        self.actor_max_grad_norm = float(getattr(params, "actor_max_grad_norm", self.max_grad_norm))
        self.critic_max_grad_norm = float(getattr(params, "critic_max_grad_norm", self.max_grad_norm))
        self.flow_actor_lr_scale = float(getattr(params, "flow_actor_lr_scale", 1.0))
        self.residual_head_lr_scale = float(getattr(params, "residual_head_lr_scale", 1.0))
        self.direct_head_lr_scale = float(getattr(params, "direct_head_lr_scale", 1.0))
        self.reset_direct_head_on_resume = bool(getattr(params, "reset_direct_head_on_resume", False))
        self.gaussian_log_std_lr_scale = float(getattr(params, "gaussian_log_std_lr_scale", 1.0))
        self.residual_head_max_grad_norm = float(
            getattr(params, "residual_head_max_grad_norm", self.actor_max_grad_norm)
        )
        self.direct_head_max_grad_norm = float(getattr(params, "direct_head_max_grad_norm", self.actor_max_grad_norm))
        self.n_cfm_samples = int(params.n_samples_per_action)
        self.cfm_diff_clip = float(params.cfm_diff_clip)
        self.cfm_diff_clip_min = getattr(params, "cfm_diff_clip_min", None)
        self.cfm_diff_clip_max = getattr(params, "cfm_diff_clip_max", None)
        self.cfm_diff_clip_min = None if self.cfm_diff_clip_min is None else float(self.cfm_diff_clip_min)
        self.cfm_diff_clip_max = None if self.cfm_diff_clip_max is None else float(self.cfm_diff_clip_max)
        self.cfm_loss_clamp = float(getattr(params, "cfm_loss_clamp", 0.0))
        clamp_neg = getattr(params, "cfm_loss_clamp_negative_advantages", None)
        self.cfm_loss_clamp_negative_advantages = None if clamp_neg is None else float(clamp_neg)
        self.log_ratio_scale = float(getattr(params, "log_ratio_scale", 1.0))
        self.trust_region_mode = str(getattr(params, "trust_region_mode", "ppo"))
        self.actor_objective = str(getattr(params, "actor_objective", "fpo")).lower()
        self.fpo_objective_coef = float(getattr(params, "fpo_objective_coef", 1.0))
        self.fpo_objective_min_coef = float(getattr(params, "fpo_objective_min_coef", self.fpo_objective_coef))
        self.fpo_objective_decay_steps = int(getattr(params, "fpo_objective_decay_steps", 0))
        self.gaussian_objective_coef = float(getattr(params, "gaussian_objective_coef", 1.0))
        self.gaussian_objective_min_coef = float(
            getattr(params, "gaussian_objective_min_coef", self.gaussian_objective_coef)
        )
        self.gaussian_objective_decay_steps = int(getattr(params, "gaussian_objective_decay_steps", 0))
        self.gaussian_action_std = getattr(params, "gaussian_action_std", 0.05)
        self.gaussian_logprob_action = str(getattr(params, "gaussian_logprob_action", "raw")).lower()
        if self.gaussian_logprob_action not in ("raw", "executed"):
            raise ValueError("agent.params.gaussian_logprob_action must be 'raw' or 'executed'")
        self.gaussian_mean_source = str(getattr(params, "gaussian_mean_source", "policy")).lower()
        if self.gaussian_mean_source not in ("policy", "ppo_base_raw", "ppo_base_native"):
            raise ValueError("agent.params.gaussian_mean_source must be 'policy', 'ppo_base_raw', or 'ppo_base_native'")
        self.gaussian_action_std_source = str(getattr(params, "gaussian_action_std_source", "fixed")).lower()
        self.gaussian_action_std_scale = float(getattr(params, "gaussian_action_std_scale", 1.0))
        self.gaussian_action_std_min_scale = float(getattr(params, "gaussian_action_std_min_scale", 0.0))
        self.gaussian_action_std_decay_steps = int(getattr(params, "gaussian_action_std_decay_steps", 0))
        self.gaussian_action_std_trainable = bool(getattr(params, "gaussian_action_std_trainable", False))
        self.gaussian_log_std = None
        self.ent_coef = float(getattr(params, "ent_coef", 0.0))
        self.spo_clip_coef = float(getattr(params, "spo_clip_coef", self.clip_range))
        self.fpo_chunk_steps = int(getattr(params, "fpo_chunk_steps", 1))
        self.average_cfm_loss_in_chunk = bool(getattr(params, "average_cfm_loss_in_chunk", False))
        self.average_cfm_loss_over_samples = bool(getattr(params, "average_cfm_loss_over_samples", False))
        self.rollout_deterministic = bool(getattr(params, "rollout_deterministic", False))
        self.rollout_action_noise_std = float(getattr(params, "rollout_action_noise_std", 0.0))
        self.positive_advantage_only = bool(getattr(params, "positive_advantage_only", False))
        self.advantage_clamp_positive = getattr(params, "advantage_clamp_positive", None)
        self.advantage_clamp_negative = getattr(params, "advantage_clamp_negative", None)
        self.advantage_clamp_positive = (
            None if self.advantage_clamp_positive is None else float(self.advantage_clamp_positive)
        )
        self.advantage_clamp_negative = (
            None if self.advantage_clamp_negative is None else float(self.advantage_clamp_negative)
        )
        self.train_actor_after_iterations = int(getattr(params, "train_actor_after_iterations", 0))
        self.normalize_advantage = bool(getattr(params, "normalize_advantage", True))
        self.use_clipped_value_loss = bool(getattr(params, "use_clipped_value_loss", False))
        self.pregrasp_threshold = float(getattr(params, "curriculum_pregrasp_threshold", 0.95))
        self.curriculum_stage1_metric = getattr(params, "curriculum_stage1_metric", None)
        self.curriculum_stage1_threshold = float(getattr(params, "curriculum_stage1_threshold", 0.0))
        self.curriculum_stage2_metric = getattr(params, "curriculum_stage2_metric", None)
        self.curriculum_stage2_threshold = float(getattr(params, "curriculum_stage2_threshold", 0.0))
        self.bc_anchor_dataset = getattr(params, "bc_anchor_dataset", None)
        self.bc_anchor_coef = float(getattr(params, "bc_anchor_coef", 0.0))
        self.bc_anchor_min_coef = float(getattr(params, "bc_anchor_min_coef", 0.0))
        self.bc_anchor_decay_steps = int(getattr(params, "bc_anchor_decay_steps", 0))
        self.bc_anchor_batch_size = int(getattr(params, "bc_anchor_batch_size", self.batch_size))
        self.action_anchor_coef = float(getattr(params, "action_anchor_coef", 0.0))
        self.action_anchor_min_coef = float(getattr(params, "action_anchor_min_coef", 0.0))
        self.action_anchor_decay_steps = int(getattr(params, "action_anchor_decay_steps", 0))
        self.action_anchor_batch_size = int(getattr(params, "action_anchor_batch_size", self.bc_anchor_batch_size))
        self.action_anchor_loss = str(getattr(params, "action_anchor_loss", "huber")).lower()
        self.action_anchor_huber_delta = float(getattr(params, "action_anchor_huber_delta", 0.05))
        self.on_policy_action_anchor_coef = float(getattr(params, "on_policy_action_anchor_coef", 0.0))
        self.on_policy_action_anchor_min_coef = float(getattr(params, "on_policy_action_anchor_min_coef", 0.0))
        self.on_policy_action_anchor_decay_steps = int(getattr(params, "on_policy_action_anchor_decay_steps", 0))
        self.obj_precision_reward_coef = float(getattr(params, "obj_precision_reward_coef", 0.0))
        self.obj_precision_reward_target = float(getattr(params, "obj_precision_reward_target", 0.001))
        self.obj_precision_reward_scale = float(getattr(params, "obj_precision_reward_scale", 250.0))
        self.obj_precision_reward_stage_min = float(getattr(params, "obj_precision_reward_stage_min", 2.0))
        self.obj_precision_reward_max = float(getattr(params, "obj_precision_reward_max", 1.0))
        self.obj_precision_penalty_coef = float(getattr(params, "obj_precision_penalty_coef", 0.0))
        self.obj_precision_penalty_target = float(getattr(params, "obj_precision_penalty_target", 0.001))
        self.obj_precision_penalty_power = float(getattr(params, "obj_precision_penalty_power", 1.0))
        self.obj_precision_penalty_max = float(getattr(params, "obj_precision_penalty_max", 1.0))
        self.obj_precision_delta_coef = float(getattr(params, "obj_precision_delta_coef", 0.0))
        self.obj_precision_delta_target = float(getattr(params, "obj_precision_delta_target", 0.0011))
        self.obj_precision_delta_max = float(getattr(params, "obj_precision_delta_max", 1.0))
        if self.obj_precision_reward_target < 0:
            raise ValueError("agent.params.obj_precision_reward_target must be non-negative")
        if self.obj_precision_reward_scale <= 0:
            raise ValueError("agent.params.obj_precision_reward_scale must be positive")
        if self.obj_precision_reward_max <= 0:
            raise ValueError("agent.params.obj_precision_reward_max must be positive")
        if self.obj_precision_penalty_target < 0:
            raise ValueError("agent.params.obj_precision_penalty_target must be non-negative")
        if self.obj_precision_penalty_coef < 0:
            raise ValueError("agent.params.obj_precision_penalty_coef must be non-negative")
        if self.obj_precision_penalty_power <= 0:
            raise ValueError("agent.params.obj_precision_penalty_power must be positive")
        if self.obj_precision_penalty_max <= 0:
            raise ValueError("agent.params.obj_precision_penalty_max must be positive")
        if self.obj_precision_delta_coef < 0:
            raise ValueError("agent.params.obj_precision_delta_coef must be non-negative")
        if self.obj_precision_delta_target < 0:
            raise ValueError("agent.params.obj_precision_delta_target must be non-negative")
        if self.obj_precision_delta_max <= 0:
            raise ValueError("agent.params.obj_precision_delta_max must be positive")
        self._last_obj_precision_err: torch.Tensor | None = None
        self.ppo_teacher_checkpoint = getattr(params, "ppo_teacher_checkpoint", None)
        self.ppo_base_checkpoint = getattr(params, "ppo_base_checkpoint", None)
        self.flow_residual_coef = float(getattr(params, "flow_residual_coef", 1.0))
        self.trainable_ppo_base_actor = bool(getattr(params, "trainable_ppo_base_actor", False))
        self.trainable_ppo_base_critic = bool(getattr(params, "trainable_ppo_base_critic", False))
        self.ppo_base_actor_lr_scale = float(getattr(params, "ppo_base_actor_lr_scale", 0.25))
        self.ppo_base_critic_lr_scale = float(getattr(params, "ppo_base_critic_lr_scale", 0.0))
        self.value_source = str(getattr(params, "value_source", "fpo")).lower()
        self.ppo_teacher_action_anchor_coef = float(getattr(params, "ppo_teacher_action_anchor_coef", 0.0))
        self.ppo_teacher_action_anchor_min_coef = float(getattr(params, "ppo_teacher_action_anchor_min_coef", 0.0))
        self.ppo_teacher_action_anchor_decay_steps = int(
            getattr(params, "ppo_teacher_action_anchor_decay_steps", 0)
        )
        self.ppo_teacher_action_anchor_deterministic = bool(
            getattr(params, "ppo_teacher_action_anchor_deterministic", True)
        )
        self.advantage_weighted_action_coef = float(getattr(params, "advantage_weighted_action_coef", 0.0))
        self.advantage_weighted_action_temp = float(getattr(params, "advantage_weighted_action_temp", 1.0))
        self.advantage_weighted_action_max_weight = float(getattr(params, "advantage_weighted_action_max_weight", 20.0))
        self.advantage_weighted_action_positive_only = bool(
            getattr(params, "advantage_weighted_action_positive_only", True)
        )
        self.advantage_weighted_action_mode = str(
            getattr(params, "advantage_weighted_action_mode", "action")
        ).lower()
        self.precision_elite_action_coef = float(getattr(params, "precision_elite_action_coef", 0.0))
        self.precision_elite_err_threshold = float(getattr(params, "precision_elite_err_threshold", 0.003))
        self.precision_elite_stage_min = float(getattr(params, "precision_elite_stage_min", 2.0))
        self.precision_elite_temp = float(getattr(params, "precision_elite_temp", 0.001))
        self.precision_elite_max_weight = float(getattr(params, "precision_elite_max_weight", 10.0))
        self.precision_elite_mode = str(getattr(params, "precision_elite_mode", "action")).lower()
        self.precision_elite_top_fraction = float(getattr(params, "precision_elite_top_fraction", 1.0))
        self.precision_elite_min_fraction = float(getattr(params, "precision_elite_min_fraction", 0.0))
        self.precision_elite_reward_quantile = float(getattr(params, "precision_elite_reward_quantile", 0.0))
        self.precision_elite_hand_threshold = getattr(params, "precision_elite_hand_threshold", None)
        self.precision_elite_control_threshold = getattr(params, "precision_elite_control_threshold", None)
        self.precision_elite_hand_threshold = (
            None if self.precision_elite_hand_threshold is None else float(self.precision_elite_hand_threshold)
        )
        self.precision_elite_control_threshold = (
            None if self.precision_elite_control_threshold is None else float(self.precision_elite_control_threshold)
        )
        self.precision_advantage_coef = float(getattr(params, "precision_advantage_coef", 0.0))
        self.precision_advantage_target = float(getattr(params, "precision_advantage_target", 0.0011))
        self.precision_advantage_threshold = float(getattr(params, "precision_advantage_threshold", 0.003))
        self.precision_advantage_stage_min = float(getattr(params, "precision_advantage_stage_min", 2.0))
        self.precision_advantage_power = float(getattr(params, "precision_advantage_power", 1.0))
        self.precision_advantage_max_bonus = float(getattr(params, "precision_advantage_max_bonus", 4.0))
        self.precision_advantage_top_fraction = float(getattr(params, "precision_advantage_top_fraction", 1.0))
        self.precision_advantage_hand_threshold = getattr(params, "precision_advantage_hand_threshold", None)
        self.precision_advantage_control_threshold = getattr(params, "precision_advantage_control_threshold", None)
        self.precision_advantage_hand_threshold = (
            None
            if self.precision_advantage_hand_threshold is None
            else float(self.precision_advantage_hand_threshold)
        )
        self.precision_advantage_control_threshold = (
            None
            if self.precision_advantage_control_threshold is None
            else float(self.precision_advantage_control_threshold)
        )
        self.self_elite_replay_coef = float(getattr(params, "self_elite_replay_coef", 0.0))
        self.self_elite_replay_capacity = int(getattr(params, "self_elite_replay_capacity", 0))
        self.self_elite_replay_batch_size = int(getattr(params, "self_elite_replay_batch_size", self.batch_size))
        self.self_elite_replay_err_threshold = float(
            getattr(params, "self_elite_replay_err_threshold", self.precision_elite_err_threshold)
        )
        self.self_elite_replay_stage_min = float(
            getattr(params, "self_elite_replay_stage_min", self.precision_elite_stage_min)
        )
        self.self_elite_replay_hand_threshold = getattr(params, "self_elite_replay_hand_threshold", None)
        self.self_elite_replay_control_threshold = getattr(params, "self_elite_replay_control_threshold", None)
        self.self_elite_replay_min_reward = getattr(params, "self_elite_replay_min_reward", None)
        self.self_elite_replay_mode = str(getattr(params, "self_elite_replay_mode", "action")).lower()
        self.self_elite_replay_weight_temp = float(getattr(params, "self_elite_replay_weight_temp", 0.001))
        self.self_elite_replay_max_weight = float(getattr(params, "self_elite_replay_max_weight", 8.0))
        self.self_elite_replay_hand_threshold = (
            None if self.self_elite_replay_hand_threshold is None else float(self.self_elite_replay_hand_threshold)
        )
        self.self_elite_replay_control_threshold = (
            None
            if self.self_elite_replay_control_threshold is None
            else float(self.self_elite_replay_control_threshold)
        )
        self.self_elite_replay_min_reward = (
            None if self.self_elite_replay_min_reward is None else float(self.self_elite_replay_min_reward)
        )
        if self.self_elite_replay_capacity < 0:
            raise ValueError("agent.params.self_elite_replay_capacity must be non-negative")
        if self.self_elite_replay_batch_size <= 0:
            raise ValueError("agent.params.self_elite_replay_batch_size must be positive")
        if self.self_elite_replay_weight_temp <= 0:
            raise ValueError("agent.params.self_elite_replay_weight_temp must be positive")
        if self.self_elite_replay_mode not in ("action", "gaussian_nll", "flow"):
            raise ValueError("agent.params.self_elite_replay_mode must be action, gaussian_nll, or flow")
        if not 0.0 < self.precision_elite_top_fraction <= 1.0:
            raise ValueError("agent.params.precision_elite_top_fraction must be in (0, 1]")
        if not 0.0 <= self.precision_elite_min_fraction <= 1.0:
            raise ValueError("agent.params.precision_elite_min_fraction must be in [0, 1]")
        if not 0.0 <= self.precision_elite_reward_quantile < 1.0:
            raise ValueError("agent.params.precision_elite_reward_quantile must be in [0, 1)")
        if self.precision_advantage_target < 0:
            raise ValueError("agent.params.precision_advantage_target must be non-negative")
        if self.precision_advantage_threshold <= self.precision_advantage_target:
            raise ValueError("agent.params.precision_advantage_threshold must be greater than target")
        if self.precision_advantage_power <= 0:
            raise ValueError("agent.params.precision_advantage_power must be positive")
        if self.precision_advantage_max_bonus <= 0:
            raise ValueError("agent.params.precision_advantage_max_bonus must be positive")
        if not 0.0 < self.precision_advantage_top_fraction <= 1.0:
            raise ValueError("agent.params.precision_advantage_top_fraction must be in (0, 1]")
        self.action_anchor_deterministic = bool(getattr(params, "action_anchor_deterministic", True))
        self.best_metric = str(getattr(params, "best_metric", "eval/mean_reward"))
        self.best_metric_mode = str(getattr(params, "best_metric_mode", "max")).lower()
        self.rfpo_precision_score_stage_min = float(getattr(params, "rfpo_precision_score_stage_min", 2.0))
        self.rfpo_precision_score_pregrasp_min = float(getattr(params, "rfpo_precision_score_pregrasp_min", 0.98))
        self.rfpo_precision_score_obj_err_target = float(
            getattr(params, "rfpo_precision_score_obj_err_target", 0.0011)
        )
        self.rfpo_precision_score_obj_err_coef = float(
            getattr(params, "rfpo_precision_score_obj_err_coef", 500.0)
        )
        self.rfpo_precision_score_obj_err_cap = float(getattr(params, "rfpo_precision_score_obj_err_cap", 2.0))
        self.save_best_model = bool(getattr(params, "save_best_model", True))
        self.rollback_to_best_on_degrade = bool(getattr(params, "rollback_to_best_on_degrade", False))
        self.rollback_patience = int(getattr(params, "rollback_patience", 0))
        self.reset_optimizer_on_rollback = bool(getattr(params, "reset_optimizer_on_rollback", True))
        self.early_stop_on_degrade = bool(getattr(params, "early_stop_on_degrade", False))
        self.early_stop_patience = int(getattr(params, "early_stop_patience", 0))
        self.degrade_metric = str(getattr(params, "degrade_metric", self.best_metric))
        self.degrade_threshold = float(getattr(params, "degrade_threshold", 0.0))
        self.target_kl = getattr(params, "target_kl", None)
        self.target_kl = None if self.target_kl is None else float(self.target_kl)
        self.max_clip_fraction = getattr(params, "max_clip_fraction", None)
        self.max_clip_fraction = None if self.max_clip_fraction is None else float(self.max_clip_fraction)
        self.lr_schedule = str(getattr(params, "lr_schedule", "fixed")).lower()
        if self.lr_schedule not in ("fixed", "adaptive"):
            raise ValueError("agent.params.lr_schedule must be 'fixed' or 'adaptive'")
        self.desired_x1_kl = float(getattr(params, "desired_x1_kl", 0.01))
        self.min_learning_rate = float(getattr(params, "min_learning_rate", 1e-7))
        self.max_learning_rate = float(getattr(params, "max_learning_rate", self.learning_rate))
        if self.max_learning_rate < self.min_learning_rate:
            raise ValueError("agent.params.max_learning_rate must be >= min_learning_rate")
        self._optimizer_lr_scales: list[float] = []
        self.phase_schedule = _normalise_phase_schedule(getattr(params, "phase_schedule", []))
        self._active_phase_index: int | None = None
        self._active_phase_name = ""
        self._active_phase_start_step = 0
        self._phase_changed_this_iter = False

        policy_config = _policy_config_from_params(params, self.obs_dim, self.action_dim)
        bc_checkpoint = getattr(params, "bc_checkpoint", None)
        if bc_checkpoint:
            checkpoint = torch.load(bc_checkpoint, map_location="cpu")
            policy_config = _merge_checkpoint_policy_config(
                checkpoint["policy_config"],
                params,
                self.obs_dim,
                self.action_dim,
            )
            if policy_config.obs_dim != self.obs_dim or policy_config.action_dim != self.action_dim:
                raise ValueError(
                    "BC checkpoint dimensions do not match environment: "
                    f"checkpoint obs/action=({policy_config.obs_dim}, {policy_config.action_dim}), "
                    f"env obs/action=({self.obs_dim}, {self.action_dim})"
                )
        self.policy = FPOStatePolicy(policy_config).to(self.device)
        if self.gaussian_action_std_trainable and self.gaussian_action_std_source == "fixed":
            self.gaussian_log_std = nn.Parameter(
                _initial_log_std(self.gaussian_action_std, self.action_dim, self.device)
            )
        self.optimizer = self._make_optimizer()
        self._record_optimizer_lr_scales()
        self.buffer = FPORolloutBuffer(
            num_envs=self.n_envs,
            num_steps=self.n_steps,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_cfm_samples=self.n_cfm_samples,
            device=self.device,
        )
        if self.fpo_chunk_steps > 1 and self.n_steps % self.fpo_chunk_steps != 0:
            raise ValueError(
                f"n_steps per env ({self.n_steps}) must be divisible by "
                f"fpo_chunk_steps ({self.fpo_chunk_steps})"
            )

        self.num_timesteps = 0
        self.iterations = 0
        self.last_obs = None
        self.current_stage = max(0, min(2, int(getattr(params, "initial_curriculum_stage", 0))))
        self.bc_anchor_obs = None
        self.bc_anchor_actions = None
        self.ppo_teacher = None
        self.ppo_base = None
        self.best_metric_value: float | None = None
        self.best_num_timesteps = 0
        self.degrade_count = 0
        self.stop_training = False
        self._initial_base_actor_params: list[torch.Tensor] = []
        self.self_elite_replay_obs: torch.Tensor | None = None
        self.self_elite_replay_actions: torch.Tensor | None = None
        self.self_elite_replay_obj_err: torch.Tensor | None = None
        self.self_elite_replay_rewards: torch.Tensor | None = None
        self._last_self_elite_replay_stats = {
            "size": 0.0,
            "added": 0.0,
            "loss": 0.0,
            "err_mean": 0.0,
            "weight_mean": 0.0,
        }
        self._last_precision_advantage_stats = {
            "frac": 0.0,
            "err_mean": 0.0,
            "bonus_mean": 0.0,
            "bonus_max": 0.0,
        }

        if bc_checkpoint:
            self.load_policy_weights(bc_checkpoint, load_optimizer=False, load_training_state=False)
        self._load_bc_anchor_dataset()
        self._load_ppo_base_and_teacher()
        self._apply_phase_schedule()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def set_logger(self, logger) -> None:
        self.logger = logger

    def _make_optimizer(self):
        flow_actor_lr_scale = float(getattr(self, "flow_actor_lr_scale", 1.0))
        residual_head_lr_scale = float(getattr(self, "residual_head_lr_scale", 1.0))
        direct_head_lr_scale = float(getattr(self, "direct_head_lr_scale", 1.0))
        gaussian_log_std_lr_scale = float(getattr(self, "gaussian_log_std_lr_scale", 1.0))
        param_groups: list[dict[str, Any]] = [
            {
                "params": list(self.policy.actor.parameters()),
                "lr": self.learning_rate * max(flow_actor_lr_scale, 0.0),
            },
            {"params": list(self.policy.critic.parameters()), "lr": self.learning_rate},
            {
                "params": list(self.policy.residual_head.parameters()),
                "lr": self.learning_rate * max(residual_head_lr_scale, 0.0),
            },
            {
                "params": list(self.policy.direct_action_head.parameters()),
                "lr": self.learning_rate * max(direct_head_lr_scale, 0.0),
            },
        ]
        if self.gaussian_log_std is not None:
            param_groups.append(
                {
                    "params": [self.gaussian_log_std],
                    "lr": self.learning_rate * max(gaussian_log_std_lr_scale, 0.0),
                }
            )
        base = getattr(self.policy, "base_policy", None)
        if base is not None and getattr(self.policy, "base_policy_trainable", False) and hasattr(base, "policy"):
            actor_params = []
            critic_params = []
            for name, param in base.policy.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("value_net") or "value_net" in name or "vf" in name:
                    critic_params.append(param)
                else:
                    actor_params.append(param)
            if actor_params:
                param_groups.append(
                    {
                        "params": actor_params,
                        "lr": self.learning_rate * max(self.ppo_base_actor_lr_scale, 0.0),
                    }
                )
            if critic_params and self.trainable_ppo_base_critic and self.ppo_base_critic_lr_scale > 0:
                param_groups.append(
                    {
                        "params": critic_params,
                        "lr": self.learning_rate * self.ppo_base_critic_lr_scale,
                    }
                )
        return torch.optim.Adam(param_groups, lr=self.learning_rate, eps=1e-5)

    def _record_optimizer_lr_scales(self) -> None:
        base_lr = max(float(self.learning_rate), 1e-12)
        self._optimizer_lr_scales = [
            float(group.get("lr", self.learning_rate)) / base_lr
            for group in self.optimizer.param_groups
        ]

    def _set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = float(np.clip(learning_rate, self.min_learning_rate, self.max_learning_rate))
        if len(self._optimizer_lr_scales) != len(self.optimizer.param_groups):
            self._record_optimizer_lr_scales()
        for group, scale in zip(self.optimizer.param_groups, self._optimizer_lr_scales):
            group["lr"] = self.learning_rate * scale

    def _apply_phase_schedule(self) -> dict[str, float]:
        self._phase_changed_this_iter = False
        if not getattr(self, "phase_schedule", None):
            return {}

        active_index: int | None = None
        active_phase: dict[str, Any] | None = None
        for index, phase in enumerate(self.phase_schedule):
            if int(getattr(self, "num_timesteps", 0)) < int(phase["start_step"]):
                break
            active_index = index
            active_phase = phase
        if active_phase is None or active_index is None:
            return {}

        phase_changed = active_index != getattr(self, "_active_phase_index", None)
        self._active_phase_index = active_index
        self._active_phase_name = str(active_phase.get("name", f"phase_{active_index}"))
        self._active_phase_start_step = int(active_phase.get("start_step", 0))
        self._phase_changed_this_iter = phase_changed

        values = dict(active_phase.get("values", {}))
        forbidden = sorted(set(values) & _FORBIDDEN_PHASE_KEYS)
        if forbidden:
            raise ValueError(f"phase_schedule cannot set warm-start fields: {', '.join(forbidden)}")

        old_lr = float(getattr(self, "learning_rate", values.get("learning_rate", 0.0)))
        for key in ("min_learning_rate", "max_learning_rate"):
            if key in values:
                setattr(self, key, float(values[key]))
        if self.max_learning_rate < self.min_learning_rate:
            raise ValueError("phase_schedule produced max_learning_rate < min_learning_rate")

        direct_assign = {
            "num_epochs": int,
            "clip_range": float,
            "max_grad_norm": float,
            "actor_max_grad_norm": float,
            "critic_max_grad_norm": float,
            "flow_actor_lr_scale": float,
            "residual_head_lr_scale": float,
            "direct_head_lr_scale": float,
            "gaussian_log_std_lr_scale": float,
            "residual_head_max_grad_norm": float,
            "direct_head_max_grad_norm": float,
            "n_cfm_samples": int,
            "cfm_diff_clip": float,
            "cfm_loss_clamp": float,
            "log_ratio_scale": float,
            "fpo_objective_coef": float,
            "fpo_objective_min_coef": float,
            "fpo_objective_decay_steps": int,
            "gaussian_objective_coef": float,
            "gaussian_objective_min_coef": float,
            "gaussian_objective_decay_steps": int,
            "gaussian_action_std_scale": float,
            "gaussian_action_std_min_scale": float,
            "gaussian_action_std_decay_steps": int,
            "ent_coef": float,
            "spo_clip_coef": float,
            "rollout_action_noise_std": float,
            "on_policy_action_anchor_coef": float,
            "on_policy_action_anchor_min_coef": float,
            "on_policy_action_anchor_decay_steps": int,
            "obj_precision_reward_coef": float,
            "obj_precision_reward_target": float,
            "obj_precision_reward_scale": float,
            "obj_precision_reward_stage_min": float,
            "obj_precision_reward_max": float,
            "obj_precision_penalty_coef": float,
            "obj_precision_penalty_target": float,
            "obj_precision_penalty_power": float,
            "obj_precision_penalty_max": float,
            "obj_precision_delta_coef": float,
            "obj_precision_delta_target": float,
            "obj_precision_delta_max": float,
            "advantage_weighted_action_coef": float,
            "advantage_weighted_action_temp": float,
            "advantage_weighted_action_max_weight": float,
            "precision_elite_action_coef": float,
            "precision_elite_err_threshold": float,
            "precision_elite_stage_min": float,
            "precision_elite_temp": float,
            "precision_elite_max_weight": float,
            "precision_elite_top_fraction": float,
            "precision_elite_min_fraction": float,
            "precision_elite_reward_quantile": float,
            "precision_advantage_coef": float,
            "precision_advantage_target": float,
            "precision_advantage_threshold": float,
            "precision_advantage_stage_min": float,
            "precision_advantage_power": float,
            "precision_advantage_max_bonus": float,
            "precision_advantage_top_fraction": float,
            "self_elite_replay_coef": float,
            "self_elite_replay_capacity": int,
            "self_elite_replay_batch_size": int,
            "self_elite_replay_err_threshold": float,
            "self_elite_replay_stage_min": float,
            "self_elite_replay_weight_temp": float,
            "self_elite_replay_max_weight": float,
            "target_kl": lambda value: None if value is None else float(value),
            "max_clip_fraction": lambda value: None if value is None else float(value),
            "degrade_threshold": float,
            "rollback_patience": int,
            "rfpo_precision_score_obj_err_coef": float,
            "rfpo_precision_score_obj_err_target": float,
            "rfpo_precision_score_obj_err_cap": float,
            "rfpo_precision_score_pregrasp_min": float,
            "rfpo_precision_score_stage_min": float,
            "curriculum_pregrasp_threshold": float,
        }
        attr_aliases = {
            "n_epochs": "num_epochs",
        }
        for key, caster in direct_assign.items():
            source_key = key
            if source_key not in values:
                yaml_key = next((alias for alias, attr in attr_aliases.items() if attr == key and alias in values), None)
                if yaml_key is None:
                    continue
                source_key = yaml_key
            setattr(self, key, caster(values[source_key]))

        string_keys = (
            "trust_region_mode",
            "actor_objective",
            "gaussian_logprob_action",
            "gaussian_mean_source",
            "gaussian_action_std_source",
            "lr_schedule",
            "advantage_weighted_action_mode",
            "precision_elite_mode",
            "self_elite_replay_mode",
            "best_metric",
            "best_metric_mode",
            "degrade_metric",
        )
        for key in string_keys:
            if key in values:
                value = str(values[key])
                if key not in ("best_metric", "degrade_metric"):
                    value = value.lower()
                setattr(self, key, value)

        bool_keys = (
            "rollout_deterministic",
            "eval_deterministic",
            "positive_advantage_only",
            "advantage_weighted_action_positive_only",
            "rollback_to_best_on_degrade",
            "reset_optimizer_on_rollback",
            "early_stop_on_degrade",
            "save_best_model",
        )
        for key in bool_keys:
            if key in values:
                setattr(self, key, bool(values[key]))

        optional_float_keys = (
            "cfm_diff_clip_min",
            "cfm_diff_clip_max",
            "cfm_loss_clamp_negative_advantages",
            "advantage_clamp_positive",
            "advantage_clamp_negative",
            "precision_elite_hand_threshold",
            "precision_elite_control_threshold",
            "precision_advantage_hand_threshold",
            "precision_advantage_control_threshold",
            "self_elite_replay_hand_threshold",
            "self_elite_replay_control_threshold",
            "self_elite_replay_min_reward",
        )
        for key in optional_float_keys:
            if key in values:
                value = values[key]
                setattr(self, key, None if value is None else float(value))

        if "gaussian_action_std" in values:
            self.gaussian_action_std = values["gaussian_action_std"]
        if "residual_action_scale" in values:
            residual_action_scale = float(values["residual_action_scale"])
            self.policy.residual_action_scale = residual_action_scale
            self.policy.cfg.residual_action_scale = residual_action_scale
        if "direct_action_scale" in values:
            direct_action_scale = float(values["direct_action_scale"])
            self.policy.direct_action_scale = direct_action_scale
            self.policy.cfg.direct_action_scale = direct_action_scale
        if "action_perturb_std" in values:
            action_perturb_std = float(values["action_perturb_std"])
            self.policy.action_perturb_std = action_perturb_std
            self.policy.cfg.action_perturb_std = action_perturb_std
        if "sampling_steps" in values:
            sampling_steps = int(values["sampling_steps"])
            self.policy.sampling_steps = sampling_steps
            self.policy.cfg.sampling_steps = sampling_steps

        should_reset_optimizer = phase_changed and bool(active_phase.get("reset_optimizer", False))
        if should_reset_optimizer:
            self.optimizer = self._make_optimizer()
            self._record_optimizer_lr_scales()
        elif "learning_rate" in values:
            self._set_learning_rate(float(values["learning_rate"]))
        elif old_lr != float(getattr(self, "learning_rate", old_lr)):
            self._set_learning_rate(float(getattr(self, "learning_rate")))

        if should_reset_optimizer and "learning_rate" in values:
            self._set_learning_rate(float(values["learning_rate"]))

        return self._phase_schedule_metrics()

    def _phase_schedule_metrics(self) -> dict[str, float]:
        return {
            "train/rfpo_phase_index": float(getattr(self, "_active_phase_index", -1) or 0),
            "train/rfpo_phase_start_step": float(getattr(self, "_active_phase_start_step", 0)),
            "train/rfpo_phase_changed": float(bool(getattr(self, "_phase_changed_this_iter", False))),
        }

    def _adjust_learning_rate_from_x1_kl(self, x1_pred_kl: torch.Tensor | float) -> float:
        kl_value = float(torch.as_tensor(x1_pred_kl).detach().cpu().item())
        if self.lr_schedule != "adaptive" or self.desired_x1_kl <= 0:
            return kl_value
        if kl_value > self.desired_x1_kl * 2.0:
            self._set_learning_rate(self.learning_rate / 1.5)
        elif 0.0 < kl_value < self.desired_x1_kl / 2.0:
            self._set_learning_rate(self.learning_rate * 1.5)
        return kl_value

    def _load_bc_anchor_dataset(self) -> None:
        needs_dataset = (
            self.bc_anchor_coef > 0
            or self.action_anchor_coef > 0
            or self.bc_anchor_min_coef > 0
            or self.action_anchor_min_coef > 0
        )
        if not self.bc_anchor_dataset or not needs_dataset:
            return
        data = np.load(self.bc_anchor_dataset)
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError("BC anchor dataset must contain rank-2 observations/actions arrays")
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("BC anchor observations and actions must have the same length")
        if observations.shape[1] != self.obs_dim or actions.shape[1] != self.action_dim:
            raise ValueError(
                "BC anchor dimensions do not match environment: "
                f"dataset obs/action=({observations.shape[1]}, {actions.shape[1]}), "
                f"env obs/action=({self.obs_dim}, {self.action_dim})"
            )
        self.bc_anchor_obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        self.bc_anchor_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

    def _load_ppo_base_and_teacher(self) -> None:
        from stable_baselines3 import PPO

        if self.ppo_base_checkpoint:
            self.ppo_base = PPO.load(self.ppo_base_checkpoint, device=self.device)
            for name, param in self.ppo_base.policy.named_parameters():
                is_critic = name.startswith("value_net") or "value_net" in name or "vf" in name
                param.requires_grad = bool(self.trainable_ppo_base_actor and (not is_critic or self.trainable_ppo_base_critic))
            self.policy.set_residual_base(
                self.ppo_base,
                residual_coef=self.flow_residual_coef,
                trainable=self.trainable_ppo_base_actor,
            )
            self._initial_base_actor_params = [
                param.detach().clone()
                for param in self.policy.trainable_base_actor_parameters()
            ]
            self.optimizer = self._make_optimizer()
            self._record_optimizer_lr_scales()

        needs_teacher = self.ppo_teacher_action_anchor_coef > 0 or self.ppo_teacher_action_anchor_min_coef > 0
        teacher_checkpoint = self.ppo_teacher_checkpoint or self.ppo_base_checkpoint
        if teacher_checkpoint and needs_teacher:
            self.ppo_teacher = PPO.load(teacher_checkpoint, device=self.device)

    def evaluate(self, observations: torch.Tensor) -> torch.Tensor:
        if self.value_source == "fpo":
            return self.policy.evaluate(observations)
        if self.value_source != "ppo_base":
            raise ValueError(f"Unknown value_source: {self.value_source}")
        if self.ppo_base is None or not hasattr(self.ppo_base, "policy"):
            raise RuntimeError("value_source=ppo_base requires agent.params.ppo_base_checkpoint")
        action_dim = int(getattr(self, "action_dim", self.policy.action_dim))
        values, _, _ = self.ppo_base.policy.evaluate_actions(
            observations,
            torch.zeros(
                observations.shape[0],
                action_dim,
                dtype=observations.dtype,
                device=observations.device,
            ),
        )
        return values

    def _current_action_std(self) -> float | torch.Tensor:
        source = str(getattr(self, "gaussian_action_std_source", "fixed")).lower()
        if source == "ppo_base" or str(self.gaussian_action_std).lower() == "ppo_base":
            if self.ppo_base is None or not hasattr(self.ppo_base, "policy") or not hasattr(self.ppo_base.policy, "log_std"):
                raise RuntimeError("gaussian_action_std_source=ppo_base requires a PPO base policy with log_std")
            device = getattr(self, "device", self.ppo_base.policy.log_std.device)
            log_std = self.ppo_base.policy.log_std
            if not getattr(self, "gaussian_action_std_trainable", False):
                log_std = log_std.detach()
            action_std = log_std.exp().to(device)
            return _scaled_action_std(
                action_std,
                scale=getattr(self, "gaussian_action_std_scale", 1.0),
                min_scale=getattr(self, "gaussian_action_std_min_scale", 0.0),
                decay_steps=getattr(self, "gaussian_action_std_decay_steps", 0),
                num_timesteps=getattr(self, "num_timesteps", 0),
            )
        if source != "fixed":
            raise ValueError(f"Unknown gaussian_action_std_source: {source}")
        if self.gaussian_log_std is not None:
            return _scaled_action_std(
                self.gaussian_log_std.exp(),
                scale=getattr(self, "gaussian_action_std_scale", 1.0),
                min_scale=getattr(self, "gaussian_action_std_min_scale", 0.0),
                decay_steps=getattr(self, "gaussian_action_std_decay_steps", 0),
                num_timesteps=getattr(self, "num_timesteps", 0),
            )
        return _scaled_action_std(
            float(self.gaussian_action_std),
            scale=getattr(self, "gaussian_action_std_scale", 1.0),
            min_scale=getattr(self, "gaussian_action_std_min_scale", 0.0),
            decay_steps=getattr(self, "gaussian_action_std_decay_steps", 0),
            num_timesteps=getattr(self, "num_timesteps", 0),
        )

    def _uses_gaussian_actor(self) -> bool:
        return self.actor_objective in ("gaussian_ppo", "hybrid_fpo")

    def _objective_weights(self) -> dict[str, float]:
        if self.actor_objective == "hybrid_fpo":
            gaussian = linear_bc_anchor_coef(
                self.gaussian_objective_coef,
                self.gaussian_objective_decay_steps,
                self.num_timesteps,
                min_coef=self.gaussian_objective_min_coef,
            )
            fpo_start = self.fpo_objective_min_coef
            fpo_end = self.fpo_objective_coef
            if self.fpo_objective_decay_steps > 0:
                progress = min(max(float(self.num_timesteps) / float(self.fpo_objective_decay_steps), 0.0), 1.0)
                fpo = fpo_start + (fpo_end - fpo_start) * progress
            else:
                fpo = fpo_end
            return {"gaussian": float(gaussian), "fpo": float(fpo)}
        if self.actor_objective == "gaussian_ppo":
            return {"gaussian": 1.0, "fpo": 0.0}
        return {"gaussian": 0.0, "fpo": 1.0}

    @staticmethod
    def _accumulate_surrogate_stats(totals: dict[str, float], prefix: str, stats: dict[str, torch.Tensor]) -> None:
        ratio = stats["ratio"].detach()
        log_ratio = stats["log_ratio"].detach()
        totals[f"train/{prefix}_ratio_mean"] += float(ratio.mean().item())
        totals[f"train/{prefix}_ratio_std"] += float(ratio.std().item())
        totals[f"train/{prefix}_ratio_min"] += float(ratio.min().item())
        totals[f"train/{prefix}_ratio_max"] += float(ratio.max().item())
        totals[f"train/{prefix}_log_ratio_mean"] += float(log_ratio.mean().item())
        totals[f"train/{prefix}_log_ratio_std"] += float(log_ratio.std().item())
        totals[f"train/{prefix}_log_ratio_min"] += float(log_ratio.min().item())
        totals[f"train/{prefix}_log_ratio_max"] += float(log_ratio.max().item())
        totals[f"train/{prefix}_clip_fraction"] += float(stats["clip_fraction"].item())
        totals[f"train/{prefix}_approx_kl"] += float(stats["approx_kl"].item())

    def _sample_anchor_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.bc_anchor_obs is None or self.bc_anchor_actions is None:
            return None
        n = self.bc_anchor_obs.shape[0]
        batch_size = min(max(1, int(batch_size)), n)
        batch_idx = torch.randint(0, n, (batch_size,), device=self.device)
        return self.bc_anchor_obs[batch_idx], self.bc_anchor_actions[batch_idx]

    def _sample_bc_anchor_loss(self, coef: float) -> torch.Tensor | None:
        if coef <= 0:
            return None
        batch = self._sample_anchor_batch(self.bc_anchor_batch_size)
        if batch is None:
            return None
        obs, actions = batch
        batch_size = obs.shape[0]
        eps, t = sample_cfm_tensors(
            batch_size=batch_size,
            n_samples=1,
            action_dim=self.action_dim,
            device=self.device,
            beta=self.policy.cfm_loss_t_inverse_cdf_beta,
        )
        bc_loss, _, _ = self.policy.get_cfm_loss(obs, actions, eps, t)
        return bc_loss.mean()

    def _action_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.action_anchor_loss == "mse":
            return F.mse_loss(predicted, target)
        if self.action_anchor_loss == "l1":
            return F.l1_loss(predicted, target)
        if self.action_anchor_loss != "huber":
            raise ValueError(f"Unknown action_anchor_loss: {self.action_anchor_loss}")
        return F.huber_loss(
            predicted,
            target,
            reduction="mean",
            delta=self.action_anchor_huber_delta,
        )

    def _sample_action_anchor_loss(self, coef: float) -> torch.Tensor | None:
        if coef <= 0:
            return None
        batch = self._sample_anchor_batch(self.action_anchor_batch_size)
        if batch is None:
            return None
        obs, actions = batch
        pred_actions = self.policy.act(obs, deterministic=self.action_anchor_deterministic)
        return self._action_loss(pred_actions, actions)

    def _on_policy_action_anchor_loss(self, batch, coef: float) -> torch.Tensor | None:
        if coef <= 0:
            return None
        pred_actions = self.policy.act(batch.obs, deterministic=True)
        return self._action_loss(pred_actions, batch.actions)

    def _ppo_teacher_action_anchor_loss(self, batch, coef: float) -> torch.Tensor | None:
        if coef <= 0 or self.ppo_teacher is None:
            return None
        obs_np = batch.obs.detach().cpu().numpy()
        teacher_actions_np, _ = self.ppo_teacher.predict(
            obs_np,
            deterministic=self.ppo_teacher_action_anchor_deterministic,
        )
        teacher_actions = torch.as_tensor(teacher_actions_np, dtype=torch.float32, device=self.device)
        pred_actions = self.policy.act(batch.obs, deterministic=True)
        return self._action_loss(pred_actions, teacher_actions)

    def _advantage_weighted_action_loss(self, batch, coef: float) -> torch.Tensor | None:
        if coef <= 0:
            return None
        advantages = batch.advantages.detach().view(-1)
        if self.advantage_weighted_action_positive_only:
            mask = advantages > 0
            if not bool(mask.any().item()):
                return None
            obs = batch.obs[mask]
            actions = batch.actions[mask]
            weights = advantages[mask]
        else:
            obs = batch.obs
            actions = batch.actions
            weights = advantages - advantages.min()
        if float(weights.sum().item()) <= 1e-8:
            return None
        temp = max(self.advantage_weighted_action_temp, 1e-6)
        weights = torch.exp((weights - weights.max()) / temp)
        weights = weights.clamp(max=self.advantage_weighted_action_max_weight)
        weights = weights / (weights.mean().clamp_min(1e-8))

        if self.advantage_weighted_action_mode == "flow":
            eps, t = sample_cfm_tensors(
                batch_size=obs.shape[0],
                n_samples=self.n_cfm_samples,
                action_dim=self.action_dim,
                device=self.device,
                beta=self.policy.cfm_loss_t_inverse_cdf_beta,
            )
            cfm_loss, _, _ = self.policy.get_cfm_loss(obs, actions, eps, t)
            per_sample = cfm_loss.mean(dim=1)
            return (weights * per_sample).mean()
        if self.advantage_weighted_action_mode != "action":
            raise ValueError(f"Unknown advantage_weighted_action_mode: {self.advantage_weighted_action_mode}")

        pred_actions = self.policy.act(obs, deterministic=True)
        per_dim = F.huber_loss(
            pred_actions,
            actions,
            reduction="none",
            delta=self.action_anchor_huber_delta,
        )
        per_sample = per_dim.mean(dim=-1)
        return (weights * per_sample).mean()

    def _precision_elite_action_loss(self, batch, coef: float) -> torch.Tensor | None:
        self._last_precision_elite_stats = {
            "frac": 0.0,
            "err_mean": 0.0,
            "hand_mean": 0.0,
            "reward_mean": 0.0,
            "weight_mean": 0.0,
        }
        if coef <= 0:
            return None
        if not hasattr(batch, "obj_com_err") or not hasattr(batch, "stage"):
            return None
        obj_err = batch.obj_com_err.detach().view(-1)
        stage = batch.stage.detach().view(-1)
        finite = torch.isfinite(obj_err)
        mask = finite & (stage >= self.precision_elite_stage_min) & (obj_err <= self.precision_elite_err_threshold)
        hand_err = getattr(batch, "hand_mjpos_err", None)
        if hand_err is not None and self.precision_elite_hand_threshold is not None:
            hand_err_flat = hand_err.detach().view(-1)
            mask = mask & torch.isfinite(hand_err_flat) & (hand_err_flat <= self.precision_elite_hand_threshold)
        else:
            hand_err_flat = None
        control_err = getattr(batch, "control_error", None)
        if control_err is not None and self.precision_elite_control_threshold is not None:
            control_err_flat = control_err.detach().view(-1)
            mask = mask & torch.isfinite(control_err_flat) & (
                control_err_flat <= self.precision_elite_control_threshold
            )
        candidate_mask = finite & (stage >= self.precision_elite_stage_min)
        if hand_err_flat is not None:
            candidate_mask = candidate_mask & torch.isfinite(hand_err_flat)
        if control_err is not None and self.precision_elite_control_threshold is not None:
            candidate_mask = candidate_mask & torch.isfinite(control_err_flat)
        env_rewards = getattr(batch, "env_rewards", None)
        reward_flat = None
        if env_rewards is not None:
            reward_flat = env_rewards.detach().view(-1)
            if self.precision_elite_reward_quantile > 0 and bool(mask.any().item()):
                masked_rewards = reward_flat[mask]
                reward_cutoff = torch.quantile(masked_rewards, self.precision_elite_reward_quantile)
                mask = mask & torch.isfinite(reward_flat) & (reward_flat >= reward_cutoff)
        precision_elite_min_fraction = float(getattr(self, "precision_elite_min_fraction", 0.0))
        min_count = 0
        if precision_elite_min_fraction > 0 and bool(candidate_mask.any().item()):
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
            min_count = max(1, int(torch.ceil(torch.tensor(
                candidate_indices.numel() * precision_elite_min_fraction,
                device=self.device,
                dtype=torch.float32,
            )).item()))
            current_count = int(mask.sum().item())
            if current_count < min_count:
                candidate_err = obj_err[candidate_indices]
                _, order = torch.topk(candidate_err, k=min(min_count, candidate_indices.numel()), largest=False)
                fallback_mask = torch.zeros_like(mask)
                fallback_mask[candidate_indices[order]] = True
                mask = mask | fallback_mask
        if not bool(mask.any().item()):
            return None
        if self.precision_elite_top_fraction < 1.0:
            masked_indices = torch.nonzero(mask, as_tuple=False).view(-1)
            keep_count = max(1, int(torch.ceil(torch.tensor(
                masked_indices.numel() * self.precision_elite_top_fraction,
                device=self.device,
                dtype=torch.float32,
            )).item()), min_count)
            keep_count = min(keep_count, masked_indices.numel())
            masked_err = obj_err[masked_indices]
            _, order = torch.topk(masked_err, k=keep_count, largest=False)
            top_mask = torch.zeros_like(mask)
            top_mask[masked_indices[order]] = True
            mask = top_mask

        obs = batch.obs.reshape(-1, self.obs_dim)[mask]
        actions = batch.actions.reshape(-1, self.action_dim)[mask]
        selected_err = obj_err[mask]
        temp = max(float(self.precision_elite_temp), 1e-8)
        weights = torch.exp(-(selected_err / temp))
        weights = weights.clamp(max=self.precision_elite_max_weight)
        weights = weights / weights.mean().clamp_min(1e-8)
        self._last_precision_elite_stats = {
            "frac": float(mask.float().mean().item()),
            "err_mean": float(selected_err.mean().item()),
            "hand_mean": (
                float(hand_err_flat[mask].mean().item())
                if hand_err_flat is not None and bool(torch.isfinite(hand_err_flat[mask]).any().item())
                else 0.0
            ),
            "reward_mean": (
                float(reward_flat[mask].mean().item())
                if reward_flat is not None and bool(torch.isfinite(reward_flat[mask]).any().item())
                else 0.0
            ),
            "weight_mean": float(weights.mean().item()),
        }

        if self.precision_elite_mode == "flow":
            eps, t = sample_cfm_tensors(
                batch_size=obs.shape[0],
                n_samples=self.n_cfm_samples,
                action_dim=self.action_dim,
                device=self.device,
                beta=self.policy.cfm_loss_t_inverse_cdf_beta,
            )
            cfm_loss, _, _ = self.policy.get_cfm_loss(obs, actions, eps, t)
            per_sample = cfm_loss.mean(dim=1)
            return (weights * per_sample).mean()
        if self.precision_elite_mode == "gaussian_nll":
            log_prob, _ = self.policy.action_log_prob(
                obs,
                actions,
                action_std=self._current_action_std(),
                use_unclipped_base_mean=self.gaussian_mean_source == "ppo_base_raw",
                use_base_policy_evaluate=self.gaussian_mean_source == "ppo_base_native",
            )
            per_sample = -log_prob.view(-1)
            return (weights * per_sample).mean()
        if self.precision_elite_mode != "action":
            raise ValueError(f"Unknown precision_elite_mode: {self.precision_elite_mode}")

        pred_actions = self.policy.act(obs, deterministic=True)
        per_dim = F.huber_loss(
            pred_actions,
            actions,
            reduction="none",
            delta=self.action_anchor_huber_delta,
        )
        per_sample = per_dim.mean(dim=-1)
        return (weights * per_sample).mean()

    def _update_self_elite_replay(self, batch) -> int:
        if self.self_elite_replay_capacity <= 0:
            return 0
        if not hasattr(batch, "obj_com_err") or not hasattr(batch, "stage"):
            return 0
        obj_err = batch.obj_com_err.detach().view(-1)
        stage = batch.stage.detach().view(-1)
        mask = (
            torch.isfinite(obj_err)
            & (stage >= self.self_elite_replay_stage_min)
            & (obj_err <= self.self_elite_replay_err_threshold)
        )
        hand_err = getattr(batch, "hand_mjpos_err", None)
        if hand_err is not None and self.self_elite_replay_hand_threshold is not None:
            hand_flat = hand_err.detach().view(-1)
            mask = mask & torch.isfinite(hand_flat) & (hand_flat <= self.self_elite_replay_hand_threshold)
        control_err = getattr(batch, "control_error", None)
        if control_err is not None and self.self_elite_replay_control_threshold is not None:
            control_flat = control_err.detach().view(-1)
            mask = mask & torch.isfinite(control_flat) & (control_flat <= self.self_elite_replay_control_threshold)
        rewards = getattr(batch, "env_rewards", None)
        reward_flat = None
        if rewards is not None:
            reward_flat = rewards.detach().view(-1)
            if self.self_elite_replay_min_reward is not None:
                mask = mask & torch.isfinite(reward_flat) & (reward_flat >= self.self_elite_replay_min_reward)
        if not bool(mask.any().item()):
            return 0

        obs = batch.obs.reshape(-1, self.obs_dim).detach()[mask]
        actions = batch.actions.reshape(-1, self.action_dim).detach()[mask]
        selected_err = obj_err[mask].detach()
        if reward_flat is None:
            selected_rewards = torch.zeros_like(selected_err)
        else:
            selected_rewards = reward_flat[mask].detach()

        if self.self_elite_replay_obs is not None:
            obs = torch.cat([self.self_elite_replay_obs, obs], dim=0)
            actions = torch.cat([self.self_elite_replay_actions, actions], dim=0)
            selected_err = torch.cat([self.self_elite_replay_obj_err, selected_err], dim=0)
            selected_rewards = torch.cat([self.self_elite_replay_rewards, selected_rewards], dim=0)

        order = torch.argsort(selected_err, stable=True)
        order = order[: self.self_elite_replay_capacity]
        self.self_elite_replay_obs = obs[order].detach()
        self.self_elite_replay_actions = actions[order].detach()
        self.self_elite_replay_obj_err = selected_err[order].detach()
        self.self_elite_replay_rewards = selected_rewards[order].detach()
        return int(mask.sum().item())

    def _self_elite_replay_loss(self, coef: float) -> torch.Tensor | None:
        self._last_self_elite_replay_stats = {
            "size": 0.0,
            "added": 0.0,
            "loss": 0.0,
            "err_mean": 0.0,
            "weight_mean": 0.0,
        }
        if coef <= 0 or self.self_elite_replay_capacity <= 0:
            return None
        if self.self_elite_replay_obs is None or self.self_elite_replay_actions is None:
            return None
        n = int(self.self_elite_replay_obs.shape[0])
        if n <= 0:
            return None
        batch_size = min(max(1, int(self.self_elite_replay_batch_size)), n)
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        obs = self.self_elite_replay_obs[idx]
        actions = self.self_elite_replay_actions[idx]
        obj_err = self.self_elite_replay_obj_err[idx]
        temp = max(float(self.self_elite_replay_weight_temp), 1e-8)
        weights = torch.exp(-(obj_err / temp)).clamp(max=self.self_elite_replay_max_weight)
        weights = weights / weights.mean().clamp_min(1e-8)

        if self.self_elite_replay_mode == "flow":
            eps, t = sample_cfm_tensors(
                batch_size=obs.shape[0],
                n_samples=self.n_cfm_samples,
                action_dim=self.action_dim,
                device=self.device,
                beta=self.policy.cfm_loss_t_inverse_cdf_beta,
            )
            cfm_loss, _, _ = self.policy.get_cfm_loss(obs, actions, eps, t)
            per_sample = cfm_loss.mean(dim=1)
        elif self.self_elite_replay_mode == "gaussian_nll":
            log_prob, _ = self.policy.action_log_prob(
                obs,
                actions,
                action_std=self._current_action_std(),
                use_unclipped_base_mean=self.gaussian_mean_source == "ppo_base_raw",
                use_base_policy_evaluate=self.gaussian_mean_source == "ppo_base_native",
            )
            per_sample = -log_prob.view(-1)
        else:
            pred_actions = self.policy.act(obs, deterministic=True)
            per_dim = F.huber_loss(
                pred_actions,
                actions,
                reduction="none",
                delta=self.action_anchor_huber_delta,
            )
            per_sample = per_dim.mean(dim=-1)
        loss = (weights * per_sample).mean()
        self._last_self_elite_replay_stats = {
            "size": float(n),
            "added": 0.0,
            "loss": float(loss.detach().item()),
            "err_mean": float(obj_err.mean().detach().item()),
            "weight_mean": float(weights.mean().detach().item()),
        }
        return loss

    def _base_actor_drift_mean(self) -> float:
        base_params = list(self.policy.trainable_base_actor_parameters())
        if not base_params or len(base_params) != len(self._initial_base_actor_params):
            return 0.0
        total = 0.0
        count = 0
        with torch.no_grad():
            for param, initial in zip(base_params, self._initial_base_actor_params):
                delta = (param.detach() - initial.to(device=param.device, dtype=param.dtype)).abs()
                total += float(delta.mean().item())
                count += 1
        return total / max(count, 1)

    def load_policy_weights(
        self,
        path: str,
        *,
        load_optimizer: bool,
        load_training_state: bool = True,
    ) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_cfg = FPOPolicyConfig.from_mapping(checkpoint["policy_config"])
        if checkpoint_cfg.obs_dim != self.obs_dim or checkpoint_cfg.action_dim != self.action_dim:
            raise ValueError(
                "Checkpoint dimensions do not match environment: "
                f"checkpoint obs/action=({checkpoint_cfg.obs_dim}, {checkpoint_cfg.action_dim}), "
                f"env obs/action=({self.obs_dim}, {self.action_dim})"
            )
        can_partially_load_old_policy = (
            checkpoint_cfg.to_dict() != self.policy.cfg.to_dict()
            and set(checkpoint["policy_state_dict"]).issubset(set(self.policy.state_dict()))
        )
        if can_partially_load_old_policy:
            incompatible = self.policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
        elif (
            checkpoint_cfg.to_dict() != self.policy.cfg.to_dict()
            and not _policy_configs_have_same_architecture(checkpoint_cfg, self.policy.cfg)
        ):
            self.policy = FPOStatePolicy(checkpoint_cfg).to(self.device)
            self.optimizer = self._make_optimizer()
            self._record_optimizer_lr_scales()
            incompatible = self.policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
        else:
            incompatible = self.policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
        allowed_missing_prefixes = ("residual_head.", "direct_action_head.")
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint is not compatible with the current FPO policy. "
                f"missing={missing}, unexpected={unexpected}"
            )
        if getattr(self, "reset_direct_head_on_resume", False):
            final_layer = self.policy.direct_action_head[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)
        normalizers = checkpoint.get("normalizers")
        if normalizers:
            self.policy.set_normalizers(
                normalizers["obs_mean"],
                normalizers["obs_std"],
                normalizers["action_mean"],
                normalizers["action_std"],
            )
        if load_optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._record_optimizer_lr_scales()
        gaussian_log_std = checkpoint.get("gaussian_log_std")
        if gaussian_log_std is not None:
            loaded = torch.as_tensor(gaussian_log_std, dtype=torch.float32, device=self.device).reshape(-1)
            if loaded.numel() != self.action_dim:
                raise ValueError(
                    f"Checkpoint gaussian_log_std has {loaded.numel()} values, expected {self.action_dim}"
            )
            self.gaussian_log_std = nn.Parameter(loaded.clone())
            self.optimizer = self._make_optimizer()
            self._record_optimizer_lr_scales()
        base_state = checkpoint.get("base_policy_state_dict")
        if base_state is not None:
            base = getattr(self.policy, "base_policy", None)
            if base is not None and hasattr(base, "policy"):
                base.policy.load_state_dict(base_state, strict=True)
        if load_training_state:
            extra = checkpoint.get("extra", {})
            replay = extra.get("self_elite_replay") if isinstance(extra, dict) else None
            if isinstance(replay, dict):
                obs = replay.get("obs")
                actions = replay.get("actions")
                obj_err = replay.get("obj_err")
                rewards = replay.get("rewards")
                if obs is not None and actions is not None and obj_err is not None:
                    self.self_elite_replay_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    self.self_elite_replay_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
                    self.self_elite_replay_obj_err = torch.as_tensor(obj_err, dtype=torch.float32, device=self.device)
                    if rewards is None:
                        self.self_elite_replay_rewards = torch.zeros_like(self.self_elite_replay_obj_err)
                    else:
                        self.self_elite_replay_rewards = torch.as_tensor(
                            rewards,
                            dtype=torch.float32,
                            device=self.device,
                        )
            self.num_timesteps = int(checkpoint.get("num_timesteps", self.num_timesteps))
            self.current_stage = int(checkpoint.get("extra", {}).get("current_stage", self.current_stage))

    def save(self, path: str) -> None:
        self_elite_replay = None
        if getattr(self, "self_elite_replay_obs", None) is not None:
            self_elite_replay = {
                "obs": self.self_elite_replay_obs.detach().cpu(),
                "actions": self.self_elite_replay_actions.detach().cpu(),
                "obj_err": self.self_elite_replay_obj_err.detach().cpu(),
                "rewards": self.self_elite_replay_rewards.detach().cpu(),
            }
        save_fpo_checkpoint(
            path=path,
            policy=self.policy,
            optimizer_state_dict=self.optimizer.state_dict(),
            num_timesteps=self.num_timesteps,
            gaussian_log_std=self.gaussian_log_std,
            extra={
                "current_stage": self.current_stage,
                "iterations": self.iterations,
                "agent_params": _params_to_dict(self.cfg.agent.params),
                "self_elite_replay": self_elite_replay,
            },
        )

    def load(self, path: str, *, load_optimizer: bool = True) -> None:
        self.load_policy_weights(path, load_optimizer=load_optimizer)

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = True):
        single_obs = False
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
            single_obs = True
        obs_tensor = torch.as_tensor(obs, device=self.device)
        was_training = self.policy.training
        self.policy.eval()
        with torch.no_grad():
            if not deterministic and self._uses_gaussian_actor():
                actions, _, _ = self.policy.sample_gaussian_action(
                    obs_tensor,
                    action_std=self._current_action_std(),
                )
            else:
                actions = self.policy.act(obs_tensor, deterministic=deterministic)
        if was_training:
            self.policy.train()
        actions_np = actions.detach().cpu().numpy()
        if single_obs:
            actions_np = actions_np[0]
        return actions_np, state

    def _object_precision_bonus_from_infos(
        self,
        infos: list[dict[str, Any]],
        dones: np.ndarray | list[bool] | None = None,
    ) -> torch.Tensor | None:
        if (
            self.obj_precision_reward_coef <= 0
            and self.obj_precision_penalty_coef <= 0
            and self.obj_precision_delta_coef <= 0
        ):
            return None
        bonuses: list[float] = []
        has_signal = False
        current_errs: list[float] = []
        for info in infos:
            shaping = 0.0
            current_err = float("inf")
            if isinstance(info, dict) and "obj_com_err" in info:
                stage = float(info.get("stage", 0.0))
                if stage >= self.obj_precision_reward_stage_min:
                    obj_err = float(info["obj_com_err"])
                    current_err = obj_err
                    bonus_err = max(obj_err - self.obj_precision_reward_target, 0.0)
                    if self.obj_precision_reward_coef > 0:
                        bonus = self.obj_precision_reward_coef * np.exp(-self.obj_precision_reward_scale * bonus_err)
                        shaping += min(float(bonus), self.obj_precision_reward_max)
                    if self.obj_precision_penalty_coef > 0:
                        penalty_err = max(obj_err - self.obj_precision_penalty_target, 0.0)
                        penalty = self.obj_precision_penalty_coef * (penalty_err ** self.obj_precision_penalty_power)
                        shaping -= min(float(penalty), self.obj_precision_penalty_max)
                    has_signal = True
            current_errs.append(current_err)
            bonuses.append(shaping)
        if self.obj_precision_delta_coef > 0 and current_errs:
            current = torch.as_tensor(current_errs, dtype=torch.float32, device=self.device)
            previous = getattr(self, "_last_obj_precision_err", None)
            if previous is not None and previous.shape == current.shape:
                finite = torch.isfinite(previous) & torch.isfinite(current)
                if bool(finite.any().item()):
                    improvement = previous - current
                    target_excess = (current - self.obj_precision_delta_target).clamp_min(0.0)
                    delta = self.obj_precision_delta_coef * improvement * (target_excess > 0).float()
                    delta = delta.clamp(-self.obj_precision_delta_max, self.obj_precision_delta_max)
                    base = torch.as_tensor(bonuses, dtype=torch.float32, device=self.device)
                    base = torch.where(finite, base + delta, base)
                    bonuses = [float(value) for value in base.detach().cpu()]
                    has_signal = True
            if previous is None or previous.shape != current.shape:
                self._last_obj_precision_err = current.detach()
            else:
                self._last_obj_precision_err = torch.where(
                    torch.isfinite(current),
                    current.detach(),
                    previous.detach(),
                )
            if dones is None:
                done_flags = [
                    bool(isinstance(info, dict) and ("terminal_observation" in info or "episode" in info))
                    for info in infos
                ]
            else:
                done_flags = [bool(done) for done in dones]
            if any(done_flags):
                done_tensor = torch.as_tensor(done_flags, dtype=torch.bool, device=self.device)
                reset_value = torch.full_like(self._last_obj_precision_err, float("inf"))
                self._last_obj_precision_err = torch.where(
                    done_tensor,
                    reset_value,
                    self._last_obj_precision_err,
                )
        if not has_signal:
            return None
        return torch.as_tensor(bonuses, dtype=torch.float32, device=self.device)

    def _object_precision_tensors_from_infos(
        self,
        infos: list[dict[str, Any]],
        env_rewards: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obj_err: list[float] = []
        hand_err: list[float] = []
        control_err: list[float] = []
        stages: list[float] = []
        for info in infos:
            if isinstance(info, dict):
                obj_err.append(float(info.get("obj_com_err", float("inf"))))
                hand_err.append(float(info.get("hand_mjpos_err", float("inf"))))
                control_err.append(float(info.get("control_error", float("inf"))))
                stages.append(float(info.get("stage", 0.0)))
            else:
                obj_err.append(float("inf"))
                hand_err.append(float("inf"))
                control_err.append(float("inf"))
                stages.append(0.0)
        if env_rewards is None:
            reward_tensor = torch.zeros(len(infos), dtype=torch.float32, device=self.device)
        else:
            reward_tensor = env_rewards.detach().to(device=self.device, dtype=torch.float32).view(-1)
        return (
            torch.as_tensor(obj_err, dtype=torch.float32, device=self.device),
            torch.as_tensor(hand_err, dtype=torch.float32, device=self.device),
            torch.as_tensor(control_err, dtype=torch.float32, device=self.device),
            reward_tensor,
            torch.as_tensor(stages, dtype=torch.float32, device=self.device),
        )

    def _record_rollout_infos(self, infos: list[dict[str, Any]]) -> dict[str, float]:
        metrics: dict[str, list[float]] = {}
        for info in infos:
            for key, value in info.items():
                if key in ("episode", "terminal_observation", "TimeLimit.truncated"):
                    continue
                if isinstance(value, (float, int, np.integer, np.floating, bool)):
                    metrics.setdefault(key, []).append(float(value))
        return {f"env/{key}": float(np.mean(values)) for key, values in metrics.items()}

    def collect_rollouts(self) -> dict[str, float]:
        self.policy.train()
        self.buffer.clear()
        done_infos: list[dict[str, Any]] = []
        if self.last_obs is None:
            self.last_obs = self.env.reset()
        precision_bonus_values: list[float] = []

        for _ in range(self.n_steps):
            obs_tensor = torch.as_tensor(self.last_obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action_std = self._current_action_std()
                if self._uses_gaussian_actor() and not self.rollout_deterministic:
                    actions, old_log_prob, _, policy_actions = self.policy.sample_gaussian_action(
                        obs_tensor,
                        action_std=action_std,
                        use_unclipped_base_mean=self.gaussian_mean_source == "ppo_base_raw",
                        use_base_policy_forward=self.gaussian_mean_source == "ppo_base_native",
                        return_raw_action=True,
                    )
                    if self.gaussian_logprob_action == "executed":
                        policy_actions = actions
                        old_log_prob, _ = self.policy.action_log_prob(
                            obs_tensor,
                            policy_actions,
                            action_std=action_std,
                            use_unclipped_base_mean=self.gaussian_mean_source == "ppo_base_raw",
                            use_base_policy_evaluate=self.gaussian_mean_source == "ppo_base_native",
                        )
                else:
                    actions = self.policy.act(obs_tensor, deterministic=self.rollout_deterministic)
                    policy_actions = actions
                    old_log_prob, _ = self.policy.action_log_prob(
                        obs_tensor,
                        policy_actions,
                        action_std=action_std,
                        use_unclipped_base_mean=self.gaussian_mean_source == "ppo_base_raw",
                        use_base_policy_evaluate=self.gaussian_mean_source == "ppo_base_native",
                    )
                if not self._uses_gaussian_actor() and self.rollout_action_noise_std > 0:
                    actions = actions + self.rollout_action_noise_std * torch.randn_like(actions)
                    if self.policy.action_clip > 0:
                        actions = actions.clamp(-self.policy.action_clip, self.policy.action_clip)
                    old_log_prob, _ = self.policy.action_log_prob(
                        obs_tensor,
                        actions,
                        action_std=action_std,
                        use_unclipped_base_mean=self.gaussian_mean_source == "ppo_base_raw",
                        use_base_policy_evaluate=self.gaussian_mean_source == "ppo_base_native",
                    )
                values = self.evaluate(obs_tensor)
                cfm_eps, cfm_t = sample_cfm_tensors(
                    batch_size=obs_tensor.shape[0],
                    n_samples=self.n_cfm_samples,
                    action_dim=self.action_dim,
                    device=self.device,
                    beta=self.policy.cfm_loss_t_inverse_cdf_beta,
                )
                flow_actions = self.policy.flow_actions_from_executed_actions(obs_tensor, actions)
                old_cfm_loss, old_x1_pred, _ = self.policy.get_cfm_loss(
                    obs_tensor,
                    flow_actions,
                    cfm_eps,
                    cfm_t,
                    actions_are_flow_targets=True,
                )

            next_obs, rewards, dones, infos = self.env.step(actions.detach().cpu().numpy())
            env_rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            rewards_tensor = env_rewards_tensor.clone()
            dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            precision_bonus = self._object_precision_bonus_from_infos(infos, dones=dones)
            if precision_bonus is not None:
                rewards_tensor = rewards_tensor + precision_bonus
                precision_bonus_values.append(float(precision_bonus.mean().detach().cpu()))
            (
                obj_com_err_tensor,
                hand_mjpos_err_tensor,
                control_error_tensor,
                env_reward_tensor,
                stage_tensor,
            ) = self._object_precision_tensors_from_infos(infos, env_rewards_tensor)

            for env_idx, info in enumerate(infos):
                if dones[env_idx]:
                    done_infos.append(info)
                terminal_obs = info.get("terminal_observation") if isinstance(info, dict) else None
                truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
                if terminal_obs is not None and truncated:
                    terminal_tensor = torch.as_tensor(
                        terminal_obs[None, :],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    with torch.no_grad():
                        terminal_value = self.evaluate(terminal_tensor).view(-1)[0]
                    rewards_tensor[env_idx] += self.gamma * terminal_value

            self.buffer.add(
                FPOTransition(
                    obs=obs_tensor.detach(),
                    actions=actions.detach(),
                    policy_actions=policy_actions.detach(),
                    flow_actions=flow_actions.detach(),
                    rewards=rewards_tensor.detach(),
                    dones=dones_tensor.detach(),
                    values=values.detach(),
                    old_cfm_loss=old_cfm_loss.detach(),
                    old_x1_pred=old_x1_pred.detach(),
                    cfm_eps=cfm_eps.detach(),
                    cfm_t=cfm_t.detach(),
                    env_rewards=env_reward_tensor.detach(),
                    old_log_prob=old_log_prob.detach(),
                    obj_com_err=obj_com_err_tensor.detach(),
                    hand_mjpos_err=hand_mjpos_err_tensor.detach(),
                    control_error=control_error_tensor.detach(),
                    stage=stage_tensor.detach(),
                )
            )
            self.last_obs = next_obs
            self.num_timesteps += self.n_envs

        last_obs_tensor = torch.as_tensor(self.last_obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            last_values = self.evaluate(last_obs_tensor)
        self.buffer.compute_returns_and_advantages(
            last_values=last_values,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize_advantage=self.normalize_advantage,
        )
        metrics = self._record_rollout_infos(done_infos)
        metrics["train/obj_precision_bonus"] = (
            float(np.mean(precision_bonus_values)) if precision_bonus_values else 0.0
        )
        return metrics

    def _actor_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        min_value = None if self.advantage_clamp_negative is None else -self.advantage_clamp_negative
        max_value = self.advantage_clamp_positive
        if min_value is not None or max_value is not None:
            advantages = advantages.clamp(min=min_value, max=max_value)
        return advantages

    def _precision_shaped_advantages(self, batch, advantages: torch.Tensor) -> torch.Tensor:
        self._last_precision_advantage_stats = {
            "frac": 0.0,
            "err_mean": 0.0,
            "bonus_mean": 0.0,
            "bonus_max": 0.0,
        }
        if self.precision_advantage_coef <= 0:
            return self._actor_advantages(advantages)
        if not hasattr(batch, "obj_com_err") or not hasattr(batch, "stage"):
            return self._actor_advantages(advantages)

        adv_shape = advantages.shape
        shaped = advantages.reshape(-1).clone()
        obj_err = batch.obj_com_err.detach().reshape(-1).to(device=shaped.device, dtype=shaped.dtype)
        stage = batch.stage.detach().reshape(-1).to(device=shaped.device, dtype=shaped.dtype)
        if obj_err.numel() != shaped.numel() or stage.numel() != shaped.numel():
            return self._actor_advantages(advantages)

        mask = (
            torch.isfinite(obj_err)
            & torch.isfinite(stage)
            & (stage >= self.precision_advantage_stage_min)
            & (obj_err <= self.precision_advantage_threshold)
        )
        valid_mask = getattr(batch, "valid_mask", None)
        if valid_mask is not None:
            valid_flat = valid_mask.detach().reshape(-1).to(device=shaped.device)
            if valid_flat.numel() == shaped.numel():
                mask = mask & valid_flat.bool()

        hand_err = getattr(batch, "hand_mjpos_err", None)
        if hand_err is not None and self.precision_advantage_hand_threshold is not None:
            hand_flat = hand_err.detach().reshape(-1).to(device=shaped.device, dtype=shaped.dtype)
            if hand_flat.numel() == shaped.numel():
                mask = mask & torch.isfinite(hand_flat) & (hand_flat <= self.precision_advantage_hand_threshold)

        control_err = getattr(batch, "control_error", None)
        if control_err is not None and self.precision_advantage_control_threshold is not None:
            control_flat = control_err.detach().reshape(-1).to(device=shaped.device, dtype=shaped.dtype)
            if control_flat.numel() == shaped.numel():
                mask = mask & torch.isfinite(control_flat) & (control_flat <= self.precision_advantage_control_threshold)

        selected = torch.nonzero(mask, as_tuple=False).view(-1)
        if selected.numel() == 0:
            return self._actor_advantages(advantages)

        if self.precision_advantage_top_fraction < 1.0:
            keep = max(1, int(float(selected.numel()) * self.precision_advantage_top_fraction))
            order = torch.argsort(obj_err[selected], stable=True)
            selected = selected[order[:keep]]

        denom = max(self.precision_advantage_threshold - self.precision_advantage_target, 1e-8)
        quality = ((self.precision_advantage_threshold - obj_err[selected]) / denom).clamp(0.0, 1.0)
        bonus = self.precision_advantage_coef * quality.pow(self.precision_advantage_power)
        bonus = bonus.clamp(max=self.precision_advantage_max_bonus)
        shaped[selected] = shaped[selected] + bonus

        self._last_precision_advantage_stats = {
            "frac": float(selected.numel()) / float(max(shaped.numel(), 1)),
            "err_mean": float(obj_err[selected].mean().detach().item()),
            "bonus_mean": float(bonus.mean().detach().item()),
            "bonus_max": float(bonus.max().detach().item()),
        }
        return self._actor_advantages(shaped.reshape(adv_shape))

    def update(self) -> dict[str, float]:
        self.policy.train()
        totals = {
            "train/surrogate_loss": 0.0,
            "train/value_loss": 0.0,
            "train/grad_norm": 0.0,
            "train/actor_grad_norm": 0.0,
            "train/base_actor_grad_norm": 0.0,
            "train/base_critic_grad_norm": 0.0,
            "train/base_actor_drift_mean": 0.0,
            "train/critic_grad_norm": 0.0,
            "train/cfm_loss": 0.0,
            "train/bc_anchor_loss": 0.0,
            "train/bc_anchor_coef": 0.0,
            "train/action_anchor_loss": 0.0,
            "train/action_anchor_coef": 0.0,
            "train/on_policy_action_anchor_loss": 0.0,
            "train/on_policy_action_anchor_coef": 0.0,
            "train/ppo_teacher_action_anchor_loss": 0.0,
            "train/ppo_teacher_action_anchor_coef": 0.0,
            "train/advantage_weighted_action_loss": 0.0,
            "train/advantage_weighted_action_coef": 0.0,
            "train/precision_elite_action_loss": 0.0,
            "train/precision_elite_action_coef": 0.0,
            "train/precision_elite_frac": 0.0,
            "train/precision_elite_err_mean": 0.0,
            "train/precision_elite_hand_mean": 0.0,
            "train/precision_elite_reward_mean": 0.0,
            "train/precision_elite_weight_mean": 0.0,
            "train/self_elite_replay_loss": 0.0,
            "train/self_elite_replay_coef": 0.0,
            "train/self_elite_replay_size": 0.0,
            "train/self_elite_replay_added": 0.0,
            "train/self_elite_replay_err_mean": 0.0,
            "train/self_elite_replay_weight_mean": 0.0,
            "train/precision_advantage_frac": 0.0,
            "train/precision_advantage_err_mean": 0.0,
            "train/precision_advantage_bonus_mean": 0.0,
            "train/precision_advantage_bonus_max": 0.0,
            "train/ratio_mean": 0.0,
            "train/ratio_std": 0.0,
            "train/ratio_min": 0.0,
            "train/ratio_max": 0.0,
            "train/log_ratio_mean": 0.0,
            "train/log_ratio_std": 0.0,
            "train/log_ratio_min": 0.0,
            "train/log_ratio_max": 0.0,
            "train/gaussian_ratio_mean": 0.0,
            "train/gaussian_ratio_std": 0.0,
            "train/gaussian_ratio_min": 0.0,
            "train/gaussian_ratio_max": 0.0,
            "train/gaussian_log_ratio_mean": 0.0,
            "train/gaussian_log_ratio_std": 0.0,
            "train/gaussian_log_ratio_min": 0.0,
            "train/gaussian_log_ratio_max": 0.0,
            "train/gaussian_clip_fraction": 0.0,
            "train/gaussian_approx_kl": 0.0,
            "train/fpo_ratio_mean": 0.0,
            "train/fpo_ratio_std": 0.0,
            "train/fpo_ratio_min": 0.0,
            "train/fpo_ratio_max": 0.0,
            "train/fpo_log_ratio_mean": 0.0,
            "train/fpo_log_ratio_std": 0.0,
            "train/fpo_log_ratio_min": 0.0,
            "train/fpo_log_ratio_max": 0.0,
            "train/fpo_clip_fraction": 0.0,
            "train/fpo_approx_kl": 0.0,
            "train/old_cfm_loss_mean": 0.0,
            "train/old_cfm_loss_std": 0.0,
            "train/new_cfm_loss_mean": 0.0,
            "train/new_cfm_loss_std": 0.0,
            "train/x1_pred_kl": 0.0,
            "train/learning_rate": 0.0,
            "train/advantage_mean": 0.0,
            "train/advantage_std": 0.0,
            "train/advantage_positive_frac": 0.0,
            "train/base_action_delta_mean": 0.0,
            "train/base_action_delta_max": 0.0,
            "train/policy_mean_update_abs": 0.0,
            "train/policy_mean_update_max": 0.0,
            "train/policy_mean_abs": 0.0,
            "train/flow_action_abs_mean": 0.0,
            "train/direct_residual_abs_mean": 0.0,
            "train/clip_fraction": 0.0,
            "train/approx_kl": 0.0,
            "train/kl_early_stop": 0.0,
            "train/clip_fraction_early_stop": 0.0,
        }
        count = 0
        if self.fpo_chunk_steps > 1:
            return self._update_chunked()

        for batch in self.buffer.iter_minibatches(
            num_minibatches=self.num_minibatches,
            num_epochs=self.num_epochs,
        ):
            with torch.no_grad():
                mean_before_update = self.policy.action_distribution_mean(batch.obs).detach()
            new_cfm_loss, x1_pred, _ = self.policy.get_cfm_loss(
                batch.obs,
                batch.flow_actions,
                batch.cfm_eps,
                batch.cfm_t,
                actions_are_flow_targets=True,
            )
            with torch.no_grad():
                x1_pred_kl = (x1_pred.detach() - batch.old_x1_pred).square().mean()
                x1_pred_kl_value = self._adjust_learning_rate_from_x1_kl(x1_pred_kl)
            new_log_prob, entropy = self.policy.action_log_prob(
                batch.obs,
                batch.policy_actions,
                action_std=self._current_action_std(),
                use_unclipped_base_mean=self.gaussian_mean_source == "ppo_base_raw",
                use_base_policy_evaluate=self.gaussian_mean_source == "ppo_base_native",
            )
            values = self.evaluate(batch.obs)
            actor_advantages = self._precision_shaped_advantages(batch, batch.advantages)
            fpo_surrogate = fpo_surrogate_loss(
                advantages=actor_advantages,
                old_cfm_loss=batch.old_cfm_loss,
                new_cfm_loss=new_cfm_loss,
                clip_range=self.clip_range,
                cfm_diff_clip=self.cfm_diff_clip,
                trust_region_mode=self.trust_region_mode,
                spo_clip_coef=self.spo_clip_coef,
                cfm_loss_clamp=self.cfm_loss_clamp,
                log_ratio_scale=self.log_ratio_scale,
                cfm_diff_clip_min=self.cfm_diff_clip_min,
                cfm_diff_clip_max=self.cfm_diff_clip_max,
                clamp_negative_advantage_loss=self.cfm_loss_clamp_negative_advantages,
                positive_advantage_only=self.positive_advantage_only,
            )
            gaussian_surrogate = gaussian_ppo_surrogate_loss(
                advantages=actor_advantages,
                old_log_prob=batch.old_log_prob,
                new_log_prob=new_log_prob,
                clip_range=self.clip_range,
            )
            objective_weights = self._objective_weights()
            if self.actor_objective == "gaussian_ppo":
                surrogate = gaussian_surrogate
            elif self.actor_objective == "hybrid_fpo":
                surrogate = {
                    **fpo_surrogate,
                    "surrogate_loss": (
                        objective_weights["fpo"] * fpo_surrogate["surrogate_loss"]
                        + objective_weights["gaussian"] * gaussian_surrogate["surrogate_loss"]
                    ),
                    "gaussian_surrogate_loss": gaussian_surrogate["surrogate_loss"],
                    "fpo_surrogate_loss": fpo_surrogate["surrogate_loss"],
                }
            else:
                surrogate = fpo_surrogate

            if self.use_clipped_value_loss:
                clipped_values = batch.old_values + (values - batch.old_values).clamp(
                    -self.clip_range,
                    self.clip_range,
                )
                value_loss = torch.max(
                    (values - batch.returns).square(),
                    (clipped_values - batch.returns).square(),
                ).mean()
            else:
                value_loss = (values - batch.returns).square().mean()

            bc_anchor_coef = linear_bc_anchor_coef(
                self.bc_anchor_coef,
                self.bc_anchor_decay_steps,
                self.num_timesteps,
                min_coef=self.bc_anchor_min_coef,
            )
            action_anchor_coef = _linear_coef_from_params(
                initial_coef=self.action_anchor_coef,
                decay_steps=self.action_anchor_decay_steps,
                num_timesteps=self.num_timesteps,
                min_coef=self.action_anchor_min_coef,
            )
            on_policy_action_anchor_coef = _linear_coef_from_params(
                initial_coef=self.on_policy_action_anchor_coef,
                decay_steps=self.on_policy_action_anchor_decay_steps,
                num_timesteps=self.num_timesteps,
                min_coef=self.on_policy_action_anchor_min_coef,
            )
            ppo_teacher_action_anchor_coef = _linear_coef_from_params(
                initial_coef=self.ppo_teacher_action_anchor_coef,
                decay_steps=self.ppo_teacher_action_anchor_decay_steps,
                num_timesteps=self.num_timesteps,
                min_coef=self.ppo_teacher_action_anchor_min_coef,
            )
            bc_anchor_loss = self._sample_bc_anchor_loss(bc_anchor_coef)
            action_anchor_loss = self._sample_action_anchor_loss(action_anchor_coef)
            on_policy_action_anchor_loss = self._on_policy_action_anchor_loss(batch, on_policy_action_anchor_coef)
            ppo_teacher_action_anchor_loss = self._ppo_teacher_action_anchor_loss(
                batch,
                ppo_teacher_action_anchor_coef,
            )
            advantage_weighted_action_loss = self._advantage_weighted_action_loss(
                batch,
                self.advantage_weighted_action_coef,
            )
            precision_elite_action_loss = self._precision_elite_action_loss(
                batch,
                self.precision_elite_action_coef,
            )
            self_elite_added = self._update_self_elite_replay(batch)
            self_elite_replay_loss = self._self_elite_replay_loss(self.self_elite_replay_coef)
            train_actor = self.iterations >= self.train_actor_after_iterations
            loss = self.vf_coef * value_loss
            if train_actor:
                loss = loss + surrogate["surrogate_loss"]
                if self.ent_coef > 0:
                    loss = loss - self.ent_coef * entropy.mean()
            if bc_anchor_loss is not None:
                loss = loss + bc_anchor_coef * bc_anchor_loss
            if train_actor and action_anchor_loss is not None:
                loss = loss + action_anchor_coef * action_anchor_loss
            if train_actor and on_policy_action_anchor_loss is not None:
                loss = loss + on_policy_action_anchor_coef * on_policy_action_anchor_loss
            if train_actor and ppo_teacher_action_anchor_loss is not None:
                loss = loss + ppo_teacher_action_anchor_coef * ppo_teacher_action_anchor_loss
            if train_actor and advantage_weighted_action_loss is not None:
                loss = loss + self.advantage_weighted_action_coef * advantage_weighted_action_loss
            if train_actor and precision_elite_action_loss is not None:
                loss = loss + self.precision_elite_action_coef * precision_elite_action_loss
            if train_actor and self_elite_replay_loss is not None:
                loss = loss + self.self_elite_replay_coef * self_elite_replay_loss
            self.optimizer.zero_grad()
            loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.actor_max_grad_norm)
            residual_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.residual_head.parameters(),
                self.residual_head_max_grad_norm,
            )
            direct_head_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.direct_action_head.parameters(),
                self.direct_head_max_grad_norm,
            )
            base_actor_params = list(self.policy.trainable_base_actor_parameters())
            if base_actor_params:
                base_actor_grad_norm = nn.utils.clip_grad_norm_(base_actor_params, self.actor_max_grad_norm)
            else:
                base_actor_grad_norm = torch.tensor(0.0, device=self.device)
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.critic_max_grad_norm)
            base_critic_params = list(self.policy.trainable_base_critic_parameters())
            if base_critic_params:
                base_critic_grad_norm = nn.utils.clip_grad_norm_(base_critic_params, self.critic_max_grad_norm)
            else:
                base_critic_grad_norm = torch.tensor(0.0, device=self.device)
            self.optimizer.step()
            with torch.no_grad():
                mean_after_update = self.policy.action_distribution_mean(batch.obs).detach()
                mean_delta = (mean_after_update - mean_before_update).abs()
                flow_component = self.policy.flow_component_action(batch.obs, deterministic=True)
                direct_residual = self.policy.direct_residual_component(batch.obs)

            totals["train/surrogate_loss"] += float(surrogate["surrogate_loss"].item())
            totals.setdefault("train/gaussian_objective_coef", 0.0)
            totals.setdefault("train/fpo_objective_coef", 0.0)
            totals.setdefault("train/gaussian_surrogate_loss", 0.0)
            totals.setdefault("train/fpo_surrogate_loss", 0.0)
            totals["train/gaussian_objective_coef"] += float(objective_weights["gaussian"])
            totals["train/fpo_objective_coef"] += float(objective_weights["fpo"])
            totals["train/gaussian_surrogate_loss"] += float(
                surrogate.get("gaussian_surrogate_loss", gaussian_surrogate["surrogate_loss"]).item()
            )
            totals["train/fpo_surrogate_loss"] += float(
                surrogate.get("fpo_surrogate_loss", fpo_surrogate["surrogate_loss"]).item()
            )
            totals.setdefault("train/train_actor", 0.0)
            totals["train/train_actor"] += float(train_actor)
            totals["train/value_loss"] += float(value_loss.item())
            totals["train/grad_norm"] += float(
                max(
                    float(actor_grad_norm),
                    float(residual_grad_norm),
                    float(direct_head_grad_norm),
                    float(base_actor_grad_norm),
                    float(critic_grad_norm),
                    float(base_critic_grad_norm),
                )
            )
            totals["train/actor_grad_norm"] += float(actor_grad_norm)
            totals.setdefault("train/residual_head_grad_norm", 0.0)
            totals.setdefault("train/direct_head_grad_norm", 0.0)
            totals["train/residual_head_grad_norm"] += float(residual_grad_norm)
            totals["train/direct_head_grad_norm"] += float(direct_head_grad_norm)
            totals["train/base_actor_grad_norm"] += float(base_actor_grad_norm)
            totals["train/base_critic_grad_norm"] += float(base_critic_grad_norm)
            totals["train/base_actor_drift_mean"] += self._base_actor_drift_mean()
            totals["train/critic_grad_norm"] += float(critic_grad_norm)
            totals["train/cfm_loss"] += float(new_cfm_loss.mean().item())
            totals["train/bc_anchor_loss"] += float(bc_anchor_loss.item()) if bc_anchor_loss is not None else 0.0
            totals["train/bc_anchor_coef"] += float(bc_anchor_coef)
            totals["train/action_anchor_loss"] += (
                float(action_anchor_loss.item()) if action_anchor_loss is not None else 0.0
            )
            totals["train/action_anchor_coef"] += float(action_anchor_coef)
            totals["train/on_policy_action_anchor_loss"] += (
                float(on_policy_action_anchor_loss.item()) if on_policy_action_anchor_loss is not None else 0.0
            )
            totals["train/on_policy_action_anchor_coef"] += float(on_policy_action_anchor_coef)
            totals["train/ppo_teacher_action_anchor_loss"] += (
                float(ppo_teacher_action_anchor_loss.item()) if ppo_teacher_action_anchor_loss is not None else 0.0
            )
            totals["train/ppo_teacher_action_anchor_coef"] += float(ppo_teacher_action_anchor_coef)
            totals["train/advantage_weighted_action_loss"] += (
                float(advantage_weighted_action_loss.item()) if advantage_weighted_action_loss is not None else 0.0
            )
            totals["train/advantage_weighted_action_coef"] += float(self.advantage_weighted_action_coef)
            totals["train/precision_elite_action_loss"] += (
                float(precision_elite_action_loss.item()) if precision_elite_action_loss is not None else 0.0
            )
            totals["train/precision_elite_action_coef"] += float(self.precision_elite_action_coef)
            precision_stats = getattr(self, "_last_precision_elite_stats", {})
            totals["train/precision_elite_frac"] += float(precision_stats.get("frac", 0.0))
            totals["train/precision_elite_err_mean"] += float(precision_stats.get("err_mean", 0.0))
            totals["train/precision_elite_hand_mean"] += float(precision_stats.get("hand_mean", 0.0))
            totals["train/precision_elite_reward_mean"] += float(precision_stats.get("reward_mean", 0.0))
            totals["train/precision_elite_weight_mean"] += float(precision_stats.get("weight_mean", 0.0))
            self_elite_stats = getattr(self, "_last_self_elite_replay_stats", {})
            totals["train/self_elite_replay_loss"] += (
                float(self_elite_replay_loss.item()) if self_elite_replay_loss is not None else 0.0
            )
            totals["train/self_elite_replay_coef"] += float(self.self_elite_replay_coef)
            totals["train/self_elite_replay_size"] += float(self_elite_stats.get("size", 0.0))
            totals["train/self_elite_replay_added"] += float(self_elite_added)
            totals["train/self_elite_replay_err_mean"] += float(self_elite_stats.get("err_mean", 0.0))
            totals["train/self_elite_replay_weight_mean"] += float(self_elite_stats.get("weight_mean", 0.0))
            precision_adv_stats = getattr(self, "_last_precision_advantage_stats", {})
            totals["train/precision_advantage_frac"] += float(precision_adv_stats.get("frac", 0.0))
            totals["train/precision_advantage_err_mean"] += float(precision_adv_stats.get("err_mean", 0.0))
            totals["train/precision_advantage_bonus_mean"] += float(precision_adv_stats.get("bonus_mean", 0.0))
            totals["train/precision_advantage_bonus_max"] += float(precision_adv_stats.get("bonus_max", 0.0))
            ratio = surrogate["ratio"].detach()
            log_ratio = surrogate["log_ratio"].detach()
            old_loss = batch.old_cfm_loss.detach()
            new_loss = new_cfm_loss.detach()
            actor_advantages = actor_advantages.detach()
            totals["train/ratio_mean"] += float(ratio.mean().item())
            totals["train/ratio_std"] += float(ratio.std().item())
            totals["train/ratio_min"] += float(ratio.min().item())
            totals["train/ratio_max"] += float(ratio.max().item())
            totals["train/log_ratio_mean"] += float(log_ratio.mean().item())
            totals["train/log_ratio_std"] += float(log_ratio.std().item())
            totals["train/log_ratio_min"] += float(log_ratio.min().item())
            totals["train/log_ratio_max"] += float(log_ratio.max().item())
            self._accumulate_surrogate_stats(totals, "gaussian", gaussian_surrogate)
            self._accumulate_surrogate_stats(totals, "fpo", fpo_surrogate)
            totals["train/old_cfm_loss_mean"] += float(old_loss.mean().item())
            totals["train/old_cfm_loss_std"] += float(old_loss.std().item())
            totals["train/new_cfm_loss_mean"] += float(new_loss.mean().item())
            totals["train/new_cfm_loss_std"] += float(new_loss.std().item())
            totals["train/x1_pred_kl"] += float(x1_pred_kl_value)
            totals["train/learning_rate"] += float(self.learning_rate)
            totals["train/advantage_mean"] += float(actor_advantages.mean().item())
            totals["train/advantage_std"] += float(actor_advantages.std().item())
            totals["train/advantage_positive_frac"] += float((actor_advantages > 0).float().mean().item())
            with torch.no_grad():
                deltas = (batch.flow_actions - batch.actions).abs()
            totals["train/base_action_delta_mean"] += float(deltas.mean().item())
            totals["train/base_action_delta_max"] += float(deltas.max().item())
            totals["train/policy_mean_update_abs"] += float(mean_delta.mean().item())
            totals["train/policy_mean_update_max"] += float(mean_delta.max().item())
            totals["train/policy_mean_abs"] += float(mean_after_update.abs().mean().item())
            totals["train/flow_action_abs_mean"] += float(flow_component.abs().mean().item())
            totals["train/direct_residual_abs_mean"] += float(direct_residual.abs().mean().item())
            totals["train/clip_fraction"] += float(surrogate["clip_fraction"].item())
            totals["train/approx_kl"] += float(surrogate["approx_kl"].item())
            count += 1
            early_stop_stats = gaussian_surrogate if self._uses_gaussian_actor() else surrogate
            if self.target_kl is not None and float(early_stop_stats["approx_kl"].item()) > self.target_kl:
                totals["train/kl_early_stop"] += 1.0
                break
            if (
                self.max_clip_fraction is not None
                and float(early_stop_stats["clip_fraction"].item()) > self.max_clip_fraction
            ):
                totals["train/clip_fraction_early_stop"] += 1.0
                break

        self.buffer.clear()
        self.iterations += 1
        denom = max(count, 1)
        return {key: value / denom for key, value in totals.items()}

    def _update_chunked(self) -> dict[str, float]:
        totals = {
            "train/surrogate_loss": 0.0,
            "train/value_loss": 0.0,
            "train/grad_norm": 0.0,
            "train/actor_grad_norm": 0.0,
            "train/base_actor_grad_norm": 0.0,
            "train/base_critic_grad_norm": 0.0,
            "train/base_actor_drift_mean": 0.0,
            "train/critic_grad_norm": 0.0,
            "train/cfm_loss": 0.0,
            "train/chunk_valid_steps": 0.0,
            "train/bc_anchor_loss": 0.0,
            "train/bc_anchor_coef": 0.0,
            "train/action_anchor_loss": 0.0,
            "train/action_anchor_coef": 0.0,
            "train/on_policy_action_anchor_loss": 0.0,
            "train/on_policy_action_anchor_coef": 0.0,
            "train/ppo_teacher_action_anchor_loss": 0.0,
            "train/ppo_teacher_action_anchor_coef": 0.0,
            "train/advantage_weighted_action_loss": 0.0,
            "train/advantage_weighted_action_coef": 0.0,
            "train/precision_elite_action_loss": 0.0,
            "train/precision_elite_action_coef": 0.0,
            "train/precision_elite_frac": 0.0,
            "train/precision_elite_err_mean": 0.0,
            "train/precision_elite_hand_mean": 0.0,
            "train/precision_elite_reward_mean": 0.0,
            "train/precision_elite_weight_mean": 0.0,
            "train/self_elite_replay_loss": 0.0,
            "train/self_elite_replay_coef": 0.0,
            "train/self_elite_replay_size": 0.0,
            "train/self_elite_replay_added": 0.0,
            "train/self_elite_replay_err_mean": 0.0,
            "train/self_elite_replay_weight_mean": 0.0,
            "train/precision_advantage_frac": 0.0,
            "train/precision_advantage_err_mean": 0.0,
            "train/precision_advantage_bonus_mean": 0.0,
            "train/precision_advantage_bonus_max": 0.0,
            "train/ratio_mean": 0.0,
            "train/ratio_std": 0.0,
            "train/ratio_min": 0.0,
            "train/ratio_max": 0.0,
            "train/log_ratio_mean": 0.0,
            "train/log_ratio_std": 0.0,
            "train/log_ratio_min": 0.0,
            "train/log_ratio_max": 0.0,
            "train/gaussian_ratio_mean": 0.0,
            "train/gaussian_ratio_std": 0.0,
            "train/gaussian_ratio_min": 0.0,
            "train/gaussian_ratio_max": 0.0,
            "train/gaussian_log_ratio_mean": 0.0,
            "train/gaussian_log_ratio_std": 0.0,
            "train/gaussian_log_ratio_min": 0.0,
            "train/gaussian_log_ratio_max": 0.0,
            "train/gaussian_clip_fraction": 0.0,
            "train/gaussian_approx_kl": 0.0,
            "train/fpo_ratio_mean": 0.0,
            "train/fpo_ratio_std": 0.0,
            "train/fpo_ratio_min": 0.0,
            "train/fpo_ratio_max": 0.0,
            "train/fpo_log_ratio_mean": 0.0,
            "train/fpo_log_ratio_std": 0.0,
            "train/fpo_log_ratio_min": 0.0,
            "train/fpo_log_ratio_max": 0.0,
            "train/fpo_clip_fraction": 0.0,
            "train/fpo_approx_kl": 0.0,
            "train/old_cfm_loss_mean": 0.0,
            "train/old_cfm_loss_std": 0.0,
            "train/new_cfm_loss_mean": 0.0,
            "train/new_cfm_loss_std": 0.0,
            "train/x1_pred_kl": 0.0,
            "train/learning_rate": 0.0,
            "train/advantage_mean": 0.0,
            "train/advantage_std": 0.0,
            "train/advantage_positive_frac": 0.0,
            "train/base_action_delta_mean": 0.0,
            "train/base_action_delta_max": 0.0,
            "train/policy_mean_update_abs": 0.0,
            "train/policy_mean_update_max": 0.0,
            "train/policy_mean_abs": 0.0,
            "train/flow_action_abs_mean": 0.0,
            "train/direct_residual_abs_mean": 0.0,
            "train/clip_fraction": 0.0,
            "train/approx_kl": 0.0,
            "train/kl_early_stop": 0.0,
            "train/clip_fraction_early_stop": 0.0,
            "train/train_actor": 0.0,
        }
        count = 0
        for batch in self.buffer.iter_chunk_minibatches(
            chunk_steps=self.fpo_chunk_steps,
            num_minibatches=self.num_minibatches,
            num_epochs=self.num_epochs,
        ):
            batch_size, chunk_steps, _ = batch.actions.shape
            flat_obs = batch.obs.reshape(batch_size * chunk_steps, self.obs_dim)
            flat_actions = batch.actions.reshape(batch_size * chunk_steps, self.action_dim)
            flat_policy_actions = batch.policy_actions.reshape(batch_size * chunk_steps, self.action_dim)
            flat_flow_actions = batch.flow_actions.reshape(batch_size * chunk_steps, self.action_dim)
            flat_obj_com_err = batch.obj_com_err.reshape(batch_size * chunk_steps, 1)
            flat_stage = batch.stage.reshape(batch_size * chunk_steps, 1)
            flat_eps = batch.cfm_eps.reshape(batch_size * chunk_steps, self.n_cfm_samples, self.action_dim)
            flat_t = batch.cfm_t.reshape(batch_size * chunk_steps, self.n_cfm_samples, 1)
            with torch.no_grad():
                mean_before_update = self.policy.action_distribution_mean(flat_obs).detach()

            new_cfm_loss, x1_pred, _ = self.policy.get_cfm_loss(
                flat_obs,
                flat_flow_actions,
                flat_eps,
                flat_t,
                actions_are_flow_targets=True,
            )
            new_cfm_loss = new_cfm_loss.reshape(batch_size, chunk_steps, self.n_cfm_samples)
            x1_pred = x1_pred.reshape(batch_size, chunk_steps, self.n_cfm_samples, self.action_dim)
            with torch.no_grad():
                valid_x1 = batch.valid_mask[:, :, None, None]
                x1_pred_kl = ((x1_pred.detach() - batch.old_x1_pred).square() * valid_x1).sum()
                x1_pred_kl = x1_pred_kl / valid_x1.sum().clamp_min(1.0) / float(self.n_cfm_samples * self.action_dim)
                x1_pred_kl_value = self._adjust_learning_rate_from_x1_kl(x1_pred_kl)
            flat_new_log_prob, flat_entropy = self.policy.action_log_prob(
                flat_obs,
                flat_policy_actions,
                action_std=self._current_action_std(),
                use_unclipped_base_mean=self.gaussian_mean_source == "ppo_base_raw",
                use_base_policy_evaluate=self.gaussian_mean_source == "ppo_base_native",
            )
            new_log_prob = flat_new_log_prob.reshape(batch_size, chunk_steps, 1)
            entropy = flat_entropy.reshape(batch_size, chunk_steps, 1)
            values = self.evaluate(flat_obs).reshape(batch_size, chunk_steps, 1)

            chunk_advantages = self._precision_shaped_advantages(batch, batch.advantages).squeeze(-1)
            valid = batch.valid_mask.unsqueeze(-1)
            old_lp = (batch.old_log_prob * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            new_lp = (new_log_prob * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            adv = (chunk_advantages * batch.valid_mask).sum(dim=1, keepdim=True)
            adv = adv / batch.valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            gaussian_surrogate = gaussian_ppo_surrogate_loss(
                advantages=adv,
                old_log_prob=old_lp,
                new_log_prob=new_lp,
                clip_range=self.clip_range,
            )
            fpo_surrogate = chunked_fpo_surrogate_loss(
                advantages=chunk_advantages,
                old_cfm_loss=batch.old_cfm_loss,
                new_cfm_loss=new_cfm_loss,
                valid_mask=batch.valid_mask,
                clip_range=self.clip_range,
                cfm_diff_clip=self.cfm_diff_clip,
                trust_region_mode=self.trust_region_mode,
                spo_clip_coef=self.spo_clip_coef,
                cfm_loss_clamp=self.cfm_loss_clamp,
                log_ratio_scale=self.log_ratio_scale,
                cfm_diff_clip_min=self.cfm_diff_clip_min,
                cfm_diff_clip_max=self.cfm_diff_clip_max,
                clamp_negative_advantage_loss=self.cfm_loss_clamp_negative_advantages,
                positive_advantage_only=self.positive_advantage_only,
                average_cfm_loss_in_chunk=self.average_cfm_loss_in_chunk,
                average_cfm_loss_over_samples=self.average_cfm_loss_over_samples,
            )
            objective_weights = self._objective_weights()
            if self.actor_objective == "gaussian_ppo":
                surrogate = gaussian_surrogate
            elif self.actor_objective == "hybrid_fpo":
                surrogate = {
                    **fpo_surrogate,
                    "surrogate_loss": (
                        objective_weights["fpo"] * fpo_surrogate["surrogate_loss"]
                        + objective_weights["gaussian"] * gaussian_surrogate["surrogate_loss"]
                    ),
                    "gaussian_surrogate_loss": gaussian_surrogate["surrogate_loss"],
                    "fpo_surrogate_loss": fpo_surrogate["surrogate_loss"],
                }
            else:
                surrogate = fpo_surrogate

            valid = batch.valid_mask.unsqueeze(-1)
            if self.use_clipped_value_loss:
                clipped_values = batch.old_values + (values - batch.old_values).clamp(
                    -self.clip_range,
                    self.clip_range,
                )
                value_loss_per_step = torch.max(
                    (values - batch.returns).square(),
                    (clipped_values - batch.returns).square(),
                )
            else:
                value_loss_per_step = (values - batch.returns).square()
            value_loss = (value_loss_per_step * valid).sum() / valid.sum().clamp_min(1.0)

            bc_anchor_coef = linear_bc_anchor_coef(
                self.bc_anchor_coef,
                self.bc_anchor_decay_steps,
                self.num_timesteps,
                min_coef=self.bc_anchor_min_coef,
            )
            action_anchor_coef = _linear_coef_from_params(
                initial_coef=self.action_anchor_coef,
                decay_steps=self.action_anchor_decay_steps,
                num_timesteps=self.num_timesteps,
                min_coef=self.action_anchor_min_coef,
            )
            on_policy_action_anchor_coef = _linear_coef_from_params(
                initial_coef=self.on_policy_action_anchor_coef,
                decay_steps=self.on_policy_action_anchor_decay_steps,
                num_timesteps=self.num_timesteps,
                min_coef=self.on_policy_action_anchor_min_coef,
            )
            ppo_teacher_action_anchor_coef = _linear_coef_from_params(
                initial_coef=self.ppo_teacher_action_anchor_coef,
                decay_steps=self.ppo_teacher_action_anchor_decay_steps,
                num_timesteps=self.num_timesteps,
                min_coef=self.ppo_teacher_action_anchor_min_coef,
            )
            flat_batch = type(
                "_FlatBatch",
                (),
                {
                    "obs": flat_obs,
                    "actions": flat_actions,
                    "obj_com_err": flat_obj_com_err,
                    "hand_mjpos_err": batch.hand_mjpos_err.reshape(batch_size * chunk_steps, 1),
                    "control_error": batch.control_error.reshape(batch_size * chunk_steps, 1),
                    "env_rewards": batch.env_rewards.reshape(batch_size * chunk_steps, 1),
                    "stage": flat_stage,
                    "advantages": batch.advantages.reshape(batch_size * chunk_steps, 1),
                },
            )()
            bc_anchor_loss = self._sample_bc_anchor_loss(bc_anchor_coef)
            action_anchor_loss = self._sample_action_anchor_loss(action_anchor_coef)
            on_policy_action_anchor_loss = self._on_policy_action_anchor_loss(flat_batch, on_policy_action_anchor_coef)
            ppo_teacher_action_anchor_loss = self._ppo_teacher_action_anchor_loss(
                flat_batch,
                ppo_teacher_action_anchor_coef,
            )
            advantage_weighted_action_loss = self._advantage_weighted_action_loss(
                flat_batch,
                self.advantage_weighted_action_coef,
            )
            precision_elite_action_loss = self._precision_elite_action_loss(
                flat_batch,
                self.precision_elite_action_coef,
            )
            self_elite_added = self._update_self_elite_replay(flat_batch)
            self_elite_replay_loss = self._self_elite_replay_loss(self.self_elite_replay_coef)

            train_actor = self.iterations >= self.train_actor_after_iterations
            loss = self.vf_coef * value_loss
            if train_actor:
                loss = loss + surrogate["surrogate_loss"]
                if self.ent_coef > 0:
                    loss = loss - self.ent_coef * ((entropy * valid).sum() / valid.sum().clamp_min(1.0))
            if bc_anchor_loss is not None:
                loss = loss + bc_anchor_coef * bc_anchor_loss
            if train_actor and action_anchor_loss is not None:
                loss = loss + action_anchor_coef * action_anchor_loss
            if train_actor and on_policy_action_anchor_loss is not None:
                loss = loss + on_policy_action_anchor_coef * on_policy_action_anchor_loss
            if train_actor and ppo_teacher_action_anchor_loss is not None:
                loss = loss + ppo_teacher_action_anchor_coef * ppo_teacher_action_anchor_loss
            if train_actor and advantage_weighted_action_loss is not None:
                loss = loss + self.advantage_weighted_action_coef * advantage_weighted_action_loss
            if train_actor and precision_elite_action_loss is not None:
                loss = loss + self.precision_elite_action_coef * precision_elite_action_loss
            if train_actor and self_elite_replay_loss is not None:
                loss = loss + self.self_elite_replay_coef * self_elite_replay_loss

            self.optimizer.zero_grad()
            loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.actor_max_grad_norm)
            residual_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.residual_head.parameters(),
                self.residual_head_max_grad_norm,
            )
            direct_head_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.direct_action_head.parameters(),
                self.direct_head_max_grad_norm,
            )
            base_actor_params = list(self.policy.trainable_base_actor_parameters())
            if base_actor_params:
                base_actor_grad_norm = nn.utils.clip_grad_norm_(base_actor_params, self.actor_max_grad_norm)
            else:
                base_actor_grad_norm = torch.tensor(0.0, device=self.device)
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.critic_max_grad_norm)
            base_critic_params = list(self.policy.trainable_base_critic_parameters())
            if base_critic_params:
                base_critic_grad_norm = nn.utils.clip_grad_norm_(base_critic_params, self.critic_max_grad_norm)
            else:
                base_critic_grad_norm = torch.tensor(0.0, device=self.device)
            self.optimizer.step()
            with torch.no_grad():
                mean_after_update = self.policy.action_distribution_mean(flat_obs).detach()
                mean_delta = (mean_after_update - mean_before_update).abs()
                flow_component = self.policy.flow_component_action(flat_obs, deterministic=True)
                direct_residual = self.policy.direct_residual_component(flat_obs)

            totals["train/surrogate_loss"] += float(surrogate["surrogate_loss"].item())
            totals.setdefault("train/gaussian_objective_coef", 0.0)
            totals.setdefault("train/fpo_objective_coef", 0.0)
            totals.setdefault("train/gaussian_surrogate_loss", 0.0)
            totals.setdefault("train/fpo_surrogate_loss", 0.0)
            totals["train/gaussian_objective_coef"] += float(objective_weights["gaussian"])
            totals["train/fpo_objective_coef"] += float(objective_weights["fpo"])
            totals["train/gaussian_surrogate_loss"] += float(
                surrogate.get("gaussian_surrogate_loss", gaussian_surrogate["surrogate_loss"]).item()
            )
            totals["train/fpo_surrogate_loss"] += float(
                surrogate.get("fpo_surrogate_loss", fpo_surrogate["surrogate_loss"]).item()
            )
            totals["train/train_actor"] += float(train_actor)
            totals["train/value_loss"] += float(value_loss.item())
            totals["train/grad_norm"] += float(
                max(
                    float(actor_grad_norm),
                    float(residual_grad_norm),
                    float(direct_head_grad_norm),
                    float(base_actor_grad_norm),
                    float(critic_grad_norm),
                    float(base_critic_grad_norm),
                )
            )
            totals["train/actor_grad_norm"] += float(actor_grad_norm)
            totals.setdefault("train/residual_head_grad_norm", 0.0)
            totals.setdefault("train/direct_head_grad_norm", 0.0)
            totals["train/residual_head_grad_norm"] += float(residual_grad_norm)
            totals["train/direct_head_grad_norm"] += float(direct_head_grad_norm)
            totals["train/base_actor_grad_norm"] += float(base_actor_grad_norm)
            totals["train/base_critic_grad_norm"] += float(base_critic_grad_norm)
            totals["train/base_actor_drift_mean"] += self._base_actor_drift_mean()
            totals["train/critic_grad_norm"] += float(critic_grad_norm)
            totals["train/cfm_loss"] += float(new_cfm_loss.mean().item())
            totals["train/chunk_valid_steps"] += float(batch.valid_mask.sum(dim=1).mean().item())
            totals["train/bc_anchor_loss"] += float(bc_anchor_loss.item()) if bc_anchor_loss is not None else 0.0
            totals["train/bc_anchor_coef"] += float(bc_anchor_coef)
            totals["train/action_anchor_loss"] += (
                float(action_anchor_loss.item()) if action_anchor_loss is not None else 0.0
            )
            totals["train/action_anchor_coef"] += float(action_anchor_coef)
            totals["train/on_policy_action_anchor_loss"] += (
                float(on_policy_action_anchor_loss.item()) if on_policy_action_anchor_loss is not None else 0.0
            )
            totals["train/on_policy_action_anchor_coef"] += float(on_policy_action_anchor_coef)
            totals["train/ppo_teacher_action_anchor_loss"] += (
                float(ppo_teacher_action_anchor_loss.item()) if ppo_teacher_action_anchor_loss is not None else 0.0
            )
            totals["train/ppo_teacher_action_anchor_coef"] += float(ppo_teacher_action_anchor_coef)
            totals["train/advantage_weighted_action_loss"] += (
                float(advantage_weighted_action_loss.item()) if advantage_weighted_action_loss is not None else 0.0
            )
            totals["train/advantage_weighted_action_coef"] += float(self.advantage_weighted_action_coef)
            totals["train/precision_elite_action_loss"] += (
                float(precision_elite_action_loss.item()) if precision_elite_action_loss is not None else 0.0
            )
            totals["train/precision_elite_action_coef"] += float(self.precision_elite_action_coef)
            precision_stats = getattr(self, "_last_precision_elite_stats", {})
            totals["train/precision_elite_frac"] += float(precision_stats.get("frac", 0.0))
            totals["train/precision_elite_err_mean"] += float(precision_stats.get("err_mean", 0.0))
            totals["train/precision_elite_hand_mean"] += float(precision_stats.get("hand_mean", 0.0))
            totals["train/precision_elite_reward_mean"] += float(precision_stats.get("reward_mean", 0.0))
            totals["train/precision_elite_weight_mean"] += float(precision_stats.get("weight_mean", 0.0))
            self_elite_stats = getattr(self, "_last_self_elite_replay_stats", {})
            totals["train/self_elite_replay_loss"] += (
                float(self_elite_replay_loss.item()) if self_elite_replay_loss is not None else 0.0
            )
            totals["train/self_elite_replay_coef"] += float(self.self_elite_replay_coef)
            totals["train/self_elite_replay_size"] += float(self_elite_stats.get("size", 0.0))
            totals["train/self_elite_replay_added"] += float(self_elite_added)
            totals["train/self_elite_replay_err_mean"] += float(self_elite_stats.get("err_mean", 0.0))
            totals["train/self_elite_replay_weight_mean"] += float(self_elite_stats.get("weight_mean", 0.0))
            precision_adv_stats = getattr(self, "_last_precision_advantage_stats", {})
            totals["train/precision_advantage_frac"] += float(precision_adv_stats.get("frac", 0.0))
            totals["train/precision_advantage_err_mean"] += float(precision_adv_stats.get("err_mean", 0.0))
            totals["train/precision_advantage_bonus_mean"] += float(precision_adv_stats.get("bonus_mean", 0.0))
            totals["train/precision_advantage_bonus_max"] += float(precision_adv_stats.get("bonus_max", 0.0))
            ratio = surrogate["ratio"].detach()
            log_ratio = surrogate["log_ratio"].detach()
            old_loss = batch.old_cfm_loss.detach()
            new_loss = new_cfm_loss.detach()
            actor_advantages = chunk_advantages.detach()
            totals["train/ratio_mean"] += float(ratio.mean().item())
            totals["train/ratio_std"] += float(ratio.std().item())
            totals["train/ratio_min"] += float(ratio.min().item())
            totals["train/ratio_max"] += float(ratio.max().item())
            totals["train/log_ratio_mean"] += float(log_ratio.mean().item())
            totals["train/log_ratio_std"] += float(log_ratio.std().item())
            totals["train/log_ratio_min"] += float(log_ratio.min().item())
            totals["train/log_ratio_max"] += float(log_ratio.max().item())
            self._accumulate_surrogate_stats(totals, "gaussian", gaussian_surrogate)
            self._accumulate_surrogate_stats(totals, "fpo", fpo_surrogate)
            totals["train/old_cfm_loss_mean"] += float(old_loss.mean().item())
            totals["train/old_cfm_loss_std"] += float(old_loss.std().item())
            totals["train/new_cfm_loss_mean"] += float(new_loss.mean().item())
            totals["train/new_cfm_loss_std"] += float(new_loss.std().item())
            totals["train/x1_pred_kl"] += float(x1_pred_kl_value)
            totals["train/learning_rate"] += float(self.learning_rate)
            totals["train/advantage_mean"] += float(actor_advantages.mean().item())
            totals["train/advantage_std"] += float(actor_advantages.std().item())
            totals["train/advantage_positive_frac"] += float((actor_advantages > 0).float().mean().item())
            with torch.no_grad():
                deltas = (flat_flow_actions - flat_actions).abs()
            totals["train/base_action_delta_mean"] += float(deltas.mean().item())
            totals["train/base_action_delta_max"] += float(deltas.max().item())
            totals["train/policy_mean_update_abs"] += float(mean_delta.mean().item())
            totals["train/policy_mean_update_max"] += float(mean_delta.max().item())
            totals["train/policy_mean_abs"] += float(mean_after_update.abs().mean().item())
            totals["train/flow_action_abs_mean"] += float(flow_component.abs().mean().item())
            totals["train/direct_residual_abs_mean"] += float(direct_residual.abs().mean().item())
            totals["train/clip_fraction"] += float(surrogate["clip_fraction"].item())
            totals["train/approx_kl"] += float(surrogate["approx_kl"].item())
            count += 1
            early_stop_stats = gaussian_surrogate if self._uses_gaussian_actor() else surrogate
            if self.target_kl is not None and float(early_stop_stats["approx_kl"].item()) > self.target_kl:
                totals["train/kl_early_stop"] += 1.0
                break
            if (
                self.max_clip_fraction is not None
                and float(early_stop_stats["clip_fraction"].item()) > self.max_clip_fraction
            ):
                totals["train/clip_fraction_early_stop"] += 1.0
                break

        self.buffer.clear()
        self.iterations += 1
        denom = max(count, 1)
        return {key: value / denom for key, value in totals.items()}

    def maybe_advance_curriculum(self, eval_metrics: dict[str, Any], eval_env) -> None:
        if should_advance_vividex_curriculum(
            current_stage=self.current_stage,
            eval_metrics=eval_metrics,
            pregrasp_threshold=self.pregrasp_threshold,
            stage1_metric=self.curriculum_stage1_metric,
            stage1_threshold=self.curriculum_stage1_threshold,
            stage2_metric=self.curriculum_stage2_metric,
            stage2_threshold=self.curriculum_stage2_threshold,
        ):
            self.current_stage += 1
            self.sync_curriculum(eval_env)

    def sync_curriculum(self, eval_env=None) -> None:
        if self.current_stage <= 0:
            return
        self.env.env_method("curriculum", self.current_stage)
        if eval_env is not None:
            eval_env.env_method("curriculum", self.current_stage)

    def _is_better_metric(self, value: float, reference: float | None) -> bool:
        if reference is None:
            return True
        if self.best_metric_mode == "min":
            return value < reference
        if self.best_metric_mode != "max":
            raise ValueError(f"Unknown best_metric_mode: {self.best_metric_mode}")
        return value > reference

    def _metric_degraded(self, value: float, reference: float | None) -> bool:
        if reference is None or self.degrade_threshold <= 0:
            return False
        if self.best_metric_mode == "min":
            return value > reference + self.degrade_threshold
        return value < reference - self.degrade_threshold

    def _rfpo_precision_score(self, eval_metrics: dict[str, Any]) -> float | None:
        reward = eval_metrics.get("eval/mean_reward")
        if reward is None:
            return None
        score = float(reward)
        stage = float(eval_metrics.get("eval/mean_stage", float("-inf")))
        pregrasp = float(eval_metrics.get("eval/mean_pregrasp_success", float("-inf")))
        stage_min = float(getattr(self, "rfpo_precision_score_stage_min", 2.0))
        pregrasp_min = float(getattr(self, "rfpo_precision_score_pregrasp_min", 0.98))
        obj_err_target = float(getattr(self, "rfpo_precision_score_obj_err_target", 0.0011))
        obj_err_coef = float(getattr(self, "rfpo_precision_score_obj_err_coef", 500.0))
        obj_err_cap = float(getattr(self, "rfpo_precision_score_obj_err_cap", 2.0))
        if stage < stage_min or pregrasp < pregrasp_min:
            score -= 100.0
        obj_err = eval_metrics.get("eval/mean_obj_com_err")
        if obj_err is None:
            score -= obj_err_cap
        else:
            excess_err = max(float(obj_err) - obj_err_target, 0.0)
            penalty = excess_err * obj_err_coef
            score -= min(penalty, obj_err_cap)
        return score

    def _handle_eval_safety(self, eval_metrics: dict[str, Any]) -> dict[str, float]:
        safety_metrics: dict[str, float] = {}
        precision_score = self._rfpo_precision_score(eval_metrics)
        if precision_score is not None:
            eval_metrics["eval/rfpo_precision_score"] = precision_score
            safety_metrics["eval/rfpo_precision_score"] = float(precision_score)
        metric_value_raw = eval_metrics.get(self.best_metric)
        if metric_value_raw is None:
            return safety_metrics

        metric_value = float(metric_value_raw)
        if self._is_better_metric(metric_value, self.best_metric_value):
            self.best_metric_value = metric_value
            self.best_num_timesteps = self.num_timesteps
            self.degrade_count = 0
            if self.save_best_model:
                self.save(os.path.join(self.output_dir, "models", "best.pt"))
        else:
            degrade_value_raw = eval_metrics.get(self.degrade_metric, metric_value)
            degraded = self._metric_degraded(float(degrade_value_raw), self.best_metric_value)
            self.degrade_count = self.degrade_count + 1 if degraded else 0

        should_stop_on_degrade = (
            self.early_stop_on_degrade
            and self.early_stop_patience > 0
            and self.degrade_count >= self.early_stop_patience
        )

        if (
            self.rollback_to_best_on_degrade
            and self.rollback_patience > 0
            and self.degrade_count >= self.rollback_patience
        ):
            best_path = os.path.join(self.output_dir, "models", "best.pt")
            if os.path.exists(best_path):
                current_num_timesteps = self.num_timesteps
                current_iterations = self.iterations
                self.load_policy_weights(best_path, load_optimizer=False)
                self.num_timesteps = current_num_timesteps
                self.iterations = current_iterations
                if self.reset_optimizer_on_rollback:
                    self.optimizer = self._make_optimizer()
                    self._record_optimizer_lr_scales()
                self.degrade_count = 0
                safety_metrics["train/rolled_back_to_best"] = 1.0

        if should_stop_on_degrade:
            self.stop_training = True
            safety_metrics["train/early_stop_triggered"] = 1.0

        if self.best_metric_value is not None:
            safety_metrics["eval/best_metric_value"] = float(self.best_metric_value)
            safety_metrics["eval/best_num_timesteps"] = float(self.best_num_timesteps)
            safety_metrics["eval/degrade_count"] = float(self.degrade_count)
        return safety_metrics

    def learn(
        self,
        *,
        total_timesteps: int,
        eval_env=None,
        eval_callback=None,
        eval_freq: int | None = None,
        save_freq: int | None = None,
        restore_freq: int | None = None,
        wandb_run=None,
    ):
        start_time = time.time()
        def next_due(freq: int | None) -> int | None:
            if freq is None:
                return None
            freq = int(freq)
            if freq <= 0:
                return None
            return ((self.num_timesteps // freq) + 1) * freq

        next_eval = next_due(eval_freq)
        next_save = next_due(save_freq)
        next_restore = next_due(restore_freq)
        self._apply_phase_schedule()
        self.sync_curriculum(eval_env)
        if eval_env is not None and eval_callback is not None and eval_freq is not None:
            eval_metrics = eval_callback(
                model=self,
                eval_env=eval_env,
                num_timesteps=self.num_timesteps,
            )
            eval_metrics.update(self._handle_eval_safety(eval_metrics))
            self.maybe_advance_curriculum(eval_metrics, eval_env)
            if self.logger is not None:
                for key, value in eval_metrics.items():
                    self.logger.record(key, value)
                self.logger.dump(self.num_timesteps)
            if wandb_run is not None:
                wandb_run.log(eval_metrics, step=self.num_timesteps)

        while self.num_timesteps < total_timesteps and not self.stop_training:
            phase_metrics = self._apply_phase_schedule()
            rollout_metrics = self.collect_rollouts()
            train_metrics = self.update()
            elapsed = max(time.time() - start_time, 1e-6)
            metrics = {
                **rollout_metrics,
                **train_metrics,
                **phase_metrics,
                "time/fps": int(self.num_timesteps / elapsed),
                "time/iterations": self.iterations,
                "time/time_elapsed": int(elapsed),
                "time/total_timesteps": self.num_timesteps,
                "train/curriculum_stage": self.current_stage,
            }

            if eval_env is not None and eval_callback is not None and next_eval is not None:
                if self.num_timesteps >= next_eval:
                    eval_metrics = eval_callback(
                        model=self,
                        eval_env=eval_env,
                        num_timesteps=self.num_timesteps,
                    )
                    eval_metrics.update(self._handle_eval_safety(eval_metrics))
                    metrics.update(eval_metrics)
                    self.maybe_advance_curriculum(eval_metrics, eval_env)
                    next_eval += int(eval_freq)

            if next_save is not None and self.num_timesteps >= next_save:
                self.save(os.path.join(self.output_dir, "models", f"fpo_step_{self.num_timesteps}.pt"))
                next_save += int(save_freq)

            if next_restore is not None and self.num_timesteps >= next_restore:
                self.save(os.path.join(self.output_dir, "restore_checkpoint.pt"))
                next_restore += int(restore_freq)

            if self.logger is not None:
                for key, value in metrics.items():
                    self.logger.record(key, value)
                self.logger.dump(self.num_timesteps)
            if wandb_run is not None:
                wandb_run.log(metrics, step=self.num_timesteps)

        self.save(os.path.join(self.output_dir, "restore_checkpoint.pt"))
        self.save(os.path.join(self.output_dir, "models", "last.pt"))
        return self
