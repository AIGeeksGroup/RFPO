from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FPOPolicyConfig:
    obs_dim: int
    action_dim: int
    actor_hidden_dims: tuple[int, ...] = (256, 128)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "tanh"
    timestep_embed_dim: int = 16
    sampling_steps: int = 8
    actor_mlp_output_scale: float = 1.0
    actor_scale: float = 1.0
    action_clip: float = 1.0
    action_perturb_std: float = 0.0
    cfm_loss_t_inverse_cdf_beta: float = 1.5
    cfm_loss_reduction: str = "mean"
    cfm_loss_use_huber: bool = True
    cfm_loss_huber_delta: float = 1.0
    cfm_loss_huber_style: str = "torch"
    flow_network_output_param: str = "u"
    cfm_loss_mode: str = "u"
    action_head_mode: str = "flow_residual"
    residual_head_hidden_dims: tuple[int, ...] = (128, 64)
    residual_action_scale: float = 0.1
    residual_head_zero_init: bool = True
    direct_head_hidden_dims: tuple[int, ...] = (256, 128)
    direct_action_scale: float = 1.0
    direct_head_zero_init: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "FPOPolicyConfig":
        clean = dict(data)
        for key in ("actor_hidden_dims", "critic_hidden_dims"):
            if key in clean and not isinstance(clean[key], tuple):
                clean[key] = tuple(clean[key])
        if "residual_head_hidden_dims" in clean and not isinstance(clean["residual_head_hidden_dims"], tuple):
            clean["residual_head_hidden_dims"] = tuple(clean["residual_head_hidden_dims"])
        if "direct_head_hidden_dims" in clean and not isinstance(clean["direct_head_hidden_dims"], tuple):
            clean["direct_head_hidden_dims"] = tuple(clean["direct_head_hidden_dims"])
        return cls(**clean)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["actor_hidden_dims"] = list(self.actor_hidden_dims)
        out["critic_hidden_dims"] = list(self.critic_hidden_dims)
        out["residual_head_hidden_dims"] = list(self.residual_head_hidden_dims)
        out["direct_head_hidden_dims"] = list(self.direct_head_hidden_dims)
        return out


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _activation(name: str) -> type[nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    if name == "gelu":
        return nn.GELU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unknown activation: {name}")


def _build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    act = _activation(activation)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(act())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _as_1d_tensor(value: torch.Tensor | Any, dim: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).view(-1)
    if tensor.numel() != dim:
        raise ValueError(f"Expected {dim} values, got {tensor.numel()}")
    return tensor


def clamp_ste(x: torch.Tensor, min: float | None = None, max: float | None = None) -> torch.Tensor:
    clamped = x.clamp(min=min, max=max)
    return x + (clamped - x).detach()


class FPOStatePolicy(nn.Module):
    is_recurrent = False

    def __init__(self, cfg: FPOPolicyConfig | dict[str, Any] | Any):
        super().__init__()
        if not isinstance(cfg, FPOPolicyConfig):
            if isinstance(cfg, dict):
                cfg = FPOPolicyConfig.from_mapping(cfg)
            else:
                cfg = FPOPolicyConfig(
                    obs_dim=int(_cfg_get(cfg, "obs_dim", _cfg_get(cfg, "num_actor_obs", 0))),
                    action_dim=int(_cfg_get(cfg, "action_dim", _cfg_get(cfg, "num_actions", 0))),
                    actor_hidden_dims=tuple(_cfg_get(cfg, "actor_hidden_dims", (256, 128))),
                    critic_hidden_dims=tuple(_cfg_get(cfg, "critic_hidden_dims", (256, 128))),
                    activation=str(_cfg_get(cfg, "activation", "tanh")),
                    timestep_embed_dim=int(_cfg_get(cfg, "timestep_embed_dim", 16)),
                    sampling_steps=int(_cfg_get(cfg, "sampling_steps", 8)),
                    actor_mlp_output_scale=float(_cfg_get(cfg, "actor_mlp_output_scale", 1.0)),
                    actor_scale=float(_cfg_get(cfg, "actor_scale", 1.0)),
                    action_clip=float(_cfg_get(cfg, "action_clip", 1.0)),
                    action_perturb_std=float(_cfg_get(cfg, "action_perturb_std", 0.0)),
                    cfm_loss_t_inverse_cdf_beta=float(_cfg_get(cfg, "cfm_loss_t_inverse_cdf_beta", 1.5)),
                    cfm_loss_reduction=str(_cfg_get(cfg, "cfm_loss_reduction", "mean")),
                    cfm_loss_use_huber=bool(_cfg_get(cfg, "cfm_loss_use_huber", True)),
                    cfm_loss_huber_delta=float(_cfg_get(cfg, "cfm_loss_huber_delta", 1.0)),
                    cfm_loss_huber_style=str(_cfg_get(cfg, "cfm_loss_huber_style", "torch")),
                    flow_network_output_param=str(_cfg_get(cfg, "flow_network_output_param", "u")),
                    cfm_loss_mode=str(_cfg_get(cfg, "cfm_loss_mode", "u")),
                    action_head_mode=str(_cfg_get(cfg, "action_head_mode", "flow_residual")),
                    residual_head_hidden_dims=tuple(_cfg_get(cfg, "residual_head_hidden_dims", (128, 64))),
                    residual_action_scale=float(_cfg_get(cfg, "residual_action_scale", 0.1)),
                    residual_head_zero_init=bool(_cfg_get(cfg, "residual_head_zero_init", True)),
                    direct_head_hidden_dims=tuple(_cfg_get(cfg, "direct_head_hidden_dims", (256, 128))),
                    direct_action_scale=float(_cfg_get(cfg, "direct_action_scale", 1.0)),
                    direct_head_zero_init=bool(_cfg_get(cfg, "direct_head_zero_init", False)),
                )

        if cfg.obs_dim <= 0 or cfg.action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")

        self.cfg = cfg
        self.obs_dim = int(cfg.obs_dim)
        self.action_dim = int(cfg.action_dim)
        self.timestep_embed_dim = int(cfg.timestep_embed_dim)
        self.sampling_steps = int(cfg.sampling_steps)
        self.actor_mlp_output_scale = float(cfg.actor_mlp_output_scale)
        self.actor_scale = float(cfg.actor_scale)
        self.action_clip = float(cfg.action_clip)
        self.action_perturb_std = float(cfg.action_perturb_std)
        self.cfm_loss_t_inverse_cdf_beta = float(cfg.cfm_loss_t_inverse_cdf_beta)
        self.cfm_loss_reduction = str(cfg.cfm_loss_reduction)
        self.cfm_loss_use_huber = bool(cfg.cfm_loss_use_huber)
        self.cfm_loss_huber_delta = float(cfg.cfm_loss_huber_delta)
        self.cfm_loss_huber_style = str(cfg.cfm_loss_huber_style).lower()
        self.flow_network_output_param = str(cfg.flow_network_output_param).lower()
        self.cfm_loss_mode = str(cfg.cfm_loss_mode).lower()
        self.action_head_mode = str(cfg.action_head_mode).lower()
        self.residual_action_scale = float(cfg.residual_action_scale)
        self.direct_action_scale = float(cfg.direct_action_scale)

        actor_input_dim = self.obs_dim + self.timestep_embed_dim + self.action_dim
        self.actor = _build_mlp(
            actor_input_dim,
            tuple(cfg.actor_hidden_dims),
            self.action_dim,
            cfg.activation,
        )
        self.critic = _build_mlp(
            self.obs_dim,
            tuple(cfg.critic_hidden_dims),
            1,
            cfg.activation,
        )
        self.residual_head = _build_mlp(
            self.obs_dim,
            tuple(cfg.residual_head_hidden_dims),
            self.action_dim,
            cfg.activation,
        )
        if cfg.residual_head_zero_init:
            final_layer = self.residual_head[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)
        self.direct_action_head = _build_mlp(
            self.obs_dim,
            tuple(cfg.direct_head_hidden_dims),
            self.action_dim,
            cfg.activation,
        )
        if cfg.direct_head_zero_init:
            final_layer = self.direct_action_head[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)

        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_std", torch.ones(self.obs_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))
        self.base_policy = None
        self.residual_coef = 1.0
        self.base_policy_trainable = False

    def reset(self, dones=None):
        return None

    def set_normalizers(
        self,
        obs_mean: torch.Tensor | Any,
        obs_std: torch.Tensor | Any,
        action_mean: torch.Tensor | Any,
        action_std: torch.Tensor | Any,
    ) -> None:
        device = self.obs_mean.device
        self.obs_mean.copy_(_as_1d_tensor(obs_mean, self.obs_dim, device))
        self.obs_std.copy_(_as_1d_tensor(obs_std, self.obs_dim, device).clamp_min(1e-6))
        self.action_mean.copy_(_as_1d_tensor(action_mean, self.action_dim, device))
        self.action_std.copy_(_as_1d_tensor(action_std, self.action_dim, device).clamp_min(1e-6))

    def normalizer_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "obs_mean": self.obs_mean.detach().cpu(),
            "obs_std": self.obs_std.detach().cpu(),
            "action_mean": self.action_mean.detach().cpu(),
            "action_std": self.action_std.detach().cpu(),
        }

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std + self.action_mean

    def _embed_timestep(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[-1] != 1:
            raise ValueError(f"Expected timestep shape (..., 1), got {tuple(t.shape)}")
        half_dim = self.timestep_embed_dim // 2
        freqs = 2 ** torch.arange(half_dim, device=t.device, dtype=t.dtype)
        embedded = torch.cat([torch.cos(t * freqs), torch.sin(t * freqs)], dim=-1)
        if self.timestep_embed_dim % 2:
            embedded = torch.cat([embedded, torch.zeros_like(embedded[..., :1])], dim=-1)
        return embedded

    def _velocity(self, obs_norm: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        actor_input = torch.cat([obs_norm, self._embed_timestep(t), x_t], dim=-1)
        return self.actor_mlp_output_scale * self.actor(actor_input)

    def _integrate_flow(self, obs_norm: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        t_path = torch.linspace(
            1.0,
            0.0,
            self.sampling_steps + 1,
            device=obs_norm.device,
            dtype=obs_norm.dtype,
        )
        batch_size = obs_norm.shape[0]
        for i in range(self.sampling_steps):
            t = t_path[i].expand(batch_size, 1)
            dt = t_path[i + 1] - t_path[i]
            x_t = x_t + dt * self._velocity(obs_norm, x_t, t)
        return x_t

    def set_residual_base(
        self,
        base_policy: Any | None,
        residual_coef: float = 1.0,
        *,
        trainable: bool = False,
    ) -> None:
        self.base_policy = base_policy
        self.residual_coef = float(residual_coef)
        self.base_policy_trainable = bool(trainable)

    def trainable_base_actor_parameters(self):
        if not self.base_policy_trainable or self.base_policy is None or not hasattr(self.base_policy, "policy"):
            return
        for name, param in self.base_policy.policy.named_parameters():
            if not param.requires_grad:
                continue
            is_critic = name.startswith("value_net") or "value_net" in name or "vf" in name
            if not is_critic:
                yield param

    def trainable_base_critic_parameters(self):
        if not self.base_policy_trainable or self.base_policy is None or not hasattr(self.base_policy, "policy"):
            return
        for name, param in self.base_policy.policy.named_parameters():
            if not param.requires_grad:
                continue
            is_critic = name.startswith("value_net") or "value_net" in name or "vf" in name
            if is_critic:
                yield param

    def actor_parameters(self, *, include_trainable_base: bool = False):
        yield from self.actor.parameters()
        yield from self.residual_head.parameters()
        yield from self.direct_action_head.parameters()
        if include_trainable_base:
            yield from self.trainable_base_actor_parameters()

    def _base_actions(self, observations: torch.Tensor, deterministic: bool) -> torch.Tensor | None:
        if self.base_policy is None:
            return None
        if self.base_policy_trainable and hasattr(self.base_policy, "policy"):
            actions, _, _ = self.base_policy.policy(observations, deterministic=deterministic)
            if self.action_clip > 0:
                actions = actions.clamp(-self.action_clip, self.action_clip)
            return actions
        obs_np = observations.detach().cpu().numpy()
        actions_np, _ = self.base_policy.predict(obs_np, deterministic=deterministic)
        return torch.as_tensor(actions_np, dtype=observations.dtype, device=observations.device)

    def _raw_base_actions(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor | None:
        if self.base_policy is None:
            return None
        if self.base_policy_trainable and hasattr(self.base_policy, "policy"):
            actions, _, _ = self.base_policy.policy(observations, deterministic=deterministic)
            return actions
        return self._base_actions(observations, deterministic=deterministic)

    def flow_actions_from_executed_actions(
        self,
        observations: torch.Tensor,
        executed_actions: torch.Tensor,
    ) -> torch.Tensor:
        base_action = self._base_actions(observations, deterministic=True)
        if base_action is None:
            if self.action_head_mode in ("flow_plus_mlp_residual", "flow_plus_direct_residual"):
                obs_norm = self.normalize_obs(observations)
                if self.action_head_mode == "flow_plus_direct_residual":
                    residual = self.direct_action_scale * torch.tanh(self.direct_action_head(obs_norm)).detach()
                else:
                    residual = self.residual_action_scale * torch.tanh(self.residual_head(obs_norm)).detach()
                flow_actions = executed_actions - residual
                if self.action_clip > 0:
                    flow_actions = flow_actions.clamp(-self.action_clip, self.action_clip)
                return flow_actions
            return executed_actions
        if self.action_head_mode in ("direct_mlp", "mlp_residual"):
            return executed_actions
        coef = max(abs(float(self.residual_coef)), 1e-6)
        residual = 0.0
        if self.action_head_mode == "flow_plus_mlp_residual":
            obs_norm = self.normalize_obs(observations)
            residual = self.residual_action_scale * torch.tanh(self.residual_head(obs_norm)).detach()
        flow_actions = base_action + (executed_actions - base_action - residual) / coef
        if self.action_clip > 0:
            flow_actions = flow_actions.clamp(-self.action_clip, self.action_clip)
        return flow_actions

    def flow_act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if observations.ndim != 2:
            raise ValueError(f"Expected observations [B, obs_dim], got {tuple(observations.shape)}")
        obs_norm = self.normalize_obs(observations)
        batch_size = observations.shape[0]
        if deterministic:
            x_t = torch.zeros(batch_size, self.action_dim, device=observations.device, dtype=observations.dtype)
        else:
            x_t = torch.randn(batch_size, self.action_dim, device=observations.device, dtype=observations.dtype)
        flow_action = self._integrate_flow(obs_norm, x_t)
        action = self.denormalize_action(self.actor_scale * flow_action)
        if self.training and not deterministic and self.action_perturb_std > 0:
            action = action + self.action_perturb_std * torch.randn_like(action)
        if self.action_clip > 0:
            action = action.clamp(-self.action_clip, self.action_clip)
        return action

    def direct_mlp_act(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2:
            raise ValueError(f"Expected observations [B, obs_dim], got {tuple(observations.shape)}")
        obs_norm = self.normalize_obs(observations)
        action = self.direct_action_scale * torch.tanh(self.direct_action_head(obs_norm))
        if self.action_clip > 0:
            action = action.clamp(-self.action_clip, self.action_clip)
        return action

    def act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if self.action_head_mode == "direct_mlp":
            return self.direct_mlp_act(observations)

        base_action = self._base_actions(observations, deterministic=True)
        if base_action is None:
            flow_action = self.flow_act(observations, deterministic=deterministic)
            if self.action_head_mode in ("flow_plus_mlp_residual", "flow_plus_direct_residual"):
                obs_norm = self.normalize_obs(observations)
                if self.action_head_mode == "flow_plus_direct_residual":
                    residual = torch.tanh(self.direct_action_head(obs_norm))
                    residual_scale = self.direct_action_scale
                else:
                    residual = torch.tanh(self.residual_head(obs_norm))
                    residual_scale = self.residual_action_scale
                action = flow_action + residual_scale * residual
                if self.action_clip > 0:
                    action = action.clamp(-self.action_clip, self.action_clip)
                return action
            return flow_action

        if self.action_head_mode == "ppo_base_only":
            action = base_action
        elif self.action_head_mode == "mlp_residual":
            obs_norm = self.normalize_obs(observations)
            residual = torch.tanh(self.residual_head(obs_norm))
            action = base_action + self.residual_action_scale * residual
        elif self.action_head_mode in ("flow_plus_mlp_residual", "flow_plus_direct_residual"):
            flow_action = self.flow_act(observations, deterministic=deterministic)
            obs_norm = self.normalize_obs(observations)
            if self.action_head_mode == "flow_plus_direct_residual":
                residual = torch.tanh(self.direct_action_head(obs_norm))
                residual_scale = self.direct_action_scale
            else:
                residual = torch.tanh(self.residual_head(obs_norm))
                residual_scale = self.residual_action_scale
            action = (
                base_action
                + self.residual_coef * (flow_action - base_action)
                + residual_scale * residual
            )
        elif self.action_head_mode == "flow_residual":
            flow_action = self.flow_act(observations, deterministic=deterministic)
            action = base_action + self.residual_coef * (flow_action - base_action)
        else:
            raise ValueError(f"Unknown action_head_mode: {self.action_head_mode}")
        if self.action_clip > 0:
            action = action.clamp(-self.action_clip, self.action_clip)
        return action

    def act_inference(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            actions = self.act(observations, deterministic=deterministic)
        if was_training:
            self.train()
        return actions

    def action_distribution_mean(
        self,
        observations: torch.Tensor,
        *,
        use_unclipped_base: bool = False,
    ) -> torch.Tensor:
        if use_unclipped_base and self.action_head_mode == "ppo_base_only":
            raw_base_action = self._raw_base_actions(observations, deterministic=True)
            if raw_base_action is not None:
                return raw_base_action
        return self.act(observations, deterministic=True)

    def flow_component_action(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.flow_act(observations, deterministic=deterministic)

    def direct_residual_component(self, observations: torch.Tensor) -> torch.Tensor:
        if self.action_head_mode != "flow_plus_direct_residual":
            return torch.zeros(
                observations.shape[0],
                self.action_dim,
                dtype=observations.dtype,
                device=observations.device,
            )
        obs_norm = self.normalize_obs(observations)
        return self.direct_action_scale * torch.tanh(self.direct_action_head(obs_norm))

    def action_log_prob(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_std: float | torch.Tensor,
        *,
        use_unclipped_base_mean: bool = False,
        use_base_policy_evaluate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            use_base_policy_evaluate
            and self.action_head_mode == "ppo_base_only"
            and self.base_policy is not None
            and hasattr(self.base_policy, "policy")
            and hasattr(self.base_policy.policy, "evaluate_actions")
        ):
            _, log_prob, entropy = self.base_policy.policy.evaluate_actions(observations, actions)
            if log_prob.ndim == 1:
                log_prob = log_prob.unsqueeze(-1)
            if entropy is None:
                entropy = torch.zeros_like(log_prob)
            elif entropy.ndim == 1:
                entropy = entropy.unsqueeze(-1)
            return log_prob, entropy
        mean = self.action_distribution_mean(
            observations,
            use_unclipped_base=use_unclipped_base_mean,
        )
        std = torch.as_tensor(action_std, dtype=mean.dtype, device=mean.device).clamp_min(1e-6)
        if std.ndim == 0:
            std = std.expand_as(mean)
        elif std.ndim == 1:
            std = std.view(1, -1).expand_as(mean)
        elif std.shape != mean.shape:
            raise ValueError(f"Bad action_std shape {tuple(std.shape)} for mean {tuple(mean.shape)}")
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def sample_gaussian_action(
        self,
        observations: torch.Tensor,
        action_std: float | torch.Tensor,
        *,
        return_raw_action: bool = False,
        use_unclipped_base_mean: bool = False,
        use_base_policy_forward: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            use_base_policy_forward
            and self.action_head_mode == "ppo_base_only"
            and self.base_policy is not None
            and hasattr(self.base_policy, "policy")
        ):
            actions, _, log_prob = self.base_policy.policy(observations, deterministic=False)
            if log_prob.ndim == 1:
                log_prob = log_prob.unsqueeze(-1)
            raw_actions = actions
            if self.action_clip > 0:
                actions = actions.clamp(-self.action_clip, self.action_clip)
            mean = actions
            if return_raw_action:
                return actions, log_prob, mean, raw_actions
            return actions, log_prob, mean
        mean = self.action_distribution_mean(
            observations,
            use_unclipped_base=use_unclipped_base_mean,
        )
        std = torch.as_tensor(action_std, dtype=mean.dtype, device=mean.device).clamp_min(1e-6)
        if std.ndim == 0:
            std = std.expand_as(mean)
        elif std.ndim == 1:
            std = std.view(1, -1).expand_as(mean)
        elif std.shape != mean.shape:
            raise ValueError(f"Bad action_std shape {tuple(std.shape)} for mean {tuple(mean.shape)}")
        dist = torch.distributions.Normal(mean, std)
        raw_actions = dist.rsample()
        actions = raw_actions
        if self.action_clip > 0:
            actions = actions.clamp(-self.action_clip, self.action_clip)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1, keepdim=True)
        if return_raw_action:
            return actions, log_prob, mean, raw_actions
        return actions, log_prob, mean

    def evaluate(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2:
            raise ValueError(f"Expected observations [B, obs_dim], got {tuple(observations.shape)}")
        return self.critic(self.normalize_obs(observations))

    def get_cfm_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        *,
        actions_are_flow_targets: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError("observations and actions must be rank-2 tensors")
        batch_size, action_dim = actions.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_dim}")
        if eps.ndim != 3 or t.ndim != 3:
            raise ValueError("eps must be [B, S, A] and t must be [B, S, 1]")
        n_samples = eps.shape[1]
        if eps.shape != (batch_size, n_samples, self.action_dim):
            raise ValueError(f"Bad eps shape: {tuple(eps.shape)}")
        if t.shape != (batch_size, n_samples, 1):
            raise ValueError(f"Bad t shape: {tuple(t.shape)}")

        obs_norm = self.normalize_obs(observations)
        if actions_are_flow_targets:
            flow_actions = actions
        else:
            flow_actions = self.flow_actions_from_executed_actions(observations, actions)
        x0 = self.normalize_action(flow_actions) / max(abs(self.actor_scale), 1e-6)
        x_t = t * eps + (1.0 - t) * x0[:, None, :]
        obs_expanded = obs_norm[:, None, :].expand(batch_size, n_samples, self.obs_dim)
        actor_input = torch.cat([obs_expanded, self._embed_timestep(t), x_t], dim=-1)
        model_out = self.actor_mlp_output_scale * self.actor(actor_input)

        t_safe = t.clamp_min(1e-5)
        if self.flow_network_output_param == "u":
            velocity_pred = model_out
            x0_pred = x_t - t * velocity_pred
            x1_pred = x0_pred + velocity_pred
        elif self.flow_network_output_param == "x0":
            x0_pred = model_out
            velocity_pred = (x_t - x0_pred) / t_safe
            x1_pred = (x_t - (1.0 - t) * x0_pred) / t_safe
        else:
            raise ValueError(f"Unknown flow_network_output_param: {self.flow_network_output_param}")

        target_velocity = eps - x0[:, None, :]
        if self.cfm_loss_mode == "u":
            loss = self._reduce_error(velocity_pred, target_velocity)
        elif self.cfm_loss_mode == "x0":
            loss = self._reduce_error(x0_pred, x0[:, None, :])
        elif self.cfm_loss_mode == "eps":
            loss = self._reduce_error(x1_pred, eps)
        else:
            raise ValueError(f"Unknown cfm_loss_mode: {self.cfm_loss_mode}")
        return loss, x1_pred, x0_pred

    def _reduce_error(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.cfm_loss_use_huber:
            if self.cfm_loss_huber_style == "fpo_control":
                diff = prediction - target
                abs_diff = diff.abs()
                delta = self.cfm_loss_huber_delta
                per_dim = torch.where(
                    abs_diff <= delta,
                    diff.square(),
                    2.0 * delta * abs_diff - delta * delta,
                )
            elif self.cfm_loss_huber_style == "torch":
                per_dim = F.huber_loss(
                    prediction,
                    target,
                    reduction="none",
                    delta=self.cfm_loss_huber_delta,
                )
            else:
                raise ValueError(f"Unknown cfm_loss_huber_style: {self.cfm_loss_huber_style}")
        else:
            per_dim = (prediction - target).square()
        if self.cfm_loss_reduction == "sum":
            return per_dim.sum(dim=-1)
        if self.cfm_loss_reduction == "sqrt":
            return per_dim.sum(dim=-1) / (per_dim.shape[-1] ** 0.5)
        if self.cfm_loss_reduction != "mean":
            raise ValueError(f"Unknown cfm_loss_reduction: {self.cfm_loss_reduction}")
        return per_dim.mean(dim=-1)


def sample_cfm_tensors(
    batch_size: int,
    n_samples: int,
    action_dim: int,
    device: torch.device,
    beta: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = torch.randn(batch_size, n_samples, action_dim, device=device)
    uniform_t = torch.rand(batch_size, n_samples, 1, device=device)
    t = 0.005 + 0.99 * (1.0 - (1.0 - uniform_t) ** (1.0 / beta))
    return eps, t


def fpo_surrogate_loss(
    *,
    advantages: torch.Tensor,
    old_cfm_loss: torch.Tensor,
    new_cfm_loss: torch.Tensor,
    clip_range: float,
    cfm_diff_clip: float,
    trust_region_mode: str = "ppo",
    spo_clip_coef: float | None = None,
    cfm_loss_clamp: float = 0.0,
    log_ratio_scale: float = 1.0,
    cfm_diff_clip_min: float | None = None,
    cfm_diff_clip_max: float | None = None,
    clamp_negative_advantage_loss: float | None = None,
    positive_advantage_only: bool = False,
    average_cfm_loss_over_samples: bool = False,
) -> dict[str, torch.Tensor]:
    if advantages.ndim == 1:
        advantages = advantages[:, None]
    if advantages.ndim != 2 or advantages.shape[1] != 1:
        raise ValueError(f"Expected advantages [B, 1], got {tuple(advantages.shape)}")
    if old_cfm_loss.shape != new_cfm_loss.shape:
        raise ValueError("old_cfm_loss and new_cfm_loss must have the same shape")
    if old_cfm_loss.shape[0] != advantages.shape[0]:
        raise ValueError("advantages and cfm losses must share batch size")
    if positive_advantage_only:
        advantages = advantages.clamp_min(0.0)

    if cfm_loss_clamp > 0:
        old_cfm_loss = clamp_ste(old_cfm_loss, max=cfm_loss_clamp)
    if clamp_negative_advantage_loss is not None:
        new_cfm_loss = torch.where(
            advantages < 0,
            new_cfm_loss.clamp(max=clamp_negative_advantage_loss),
            new_cfm_loss,
        )
    if average_cfm_loss_over_samples:
        old_cfm_loss = old_cfm_loss.mean(dim=-1, keepdim=True)
        new_cfm_loss = new_cfm_loss.mean(dim=-1, keepdim=True)

    log_ratio = float(log_ratio_scale) * (old_cfm_loss - new_cfm_loss)
    if cfm_diff_clip_min is None and cfm_diff_clip_max is None and cfm_diff_clip > 0:
        cfm_diff_clip_min = -cfm_diff_clip
        cfm_diff_clip_max = cfm_diff_clip
    if cfm_diff_clip_min is not None or cfm_diff_clip_max is not None:
        log_ratio = clamp_ste(log_ratio, min=cfm_diff_clip_min, max=cfm_diff_clip_max)
    ratio = torch.exp(log_ratio)

    mode = trust_region_mode.lower()
    if mode == "ppo":
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        surrogate_loss = torch.max(pg_loss1, pg_loss2).mean()
    elif mode == "spo":
        coef = clip_range if spo_clip_coef is None else spo_clip_coef
        obj = advantages * ratio - advantages.abs() / (2.0 * max(coef, 1e-8)) * (ratio - 1.0).square()
        surrogate_loss = -obj.mean()
    elif mode == "aspo":
        coef = clip_range if spo_clip_coef is None else spo_clip_coef
        ppo_loss = torch.max(
            -advantages * ratio,
            -advantages * ratio.clamp(1.0 - clip_range, 1.0 + clip_range),
        )
        spo_loss = -(
            advantages * ratio
            - advantages.abs() / (2.0 * max(coef, 1e-8)) * (ratio - 1.0).square()
        )
        surrogate_loss = torch.where(advantages > 0, ppo_loss, spo_loss).mean()
    else:
        raise ValueError(f"Unknown trust_region_mode: {trust_region_mode}")

    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
    return {
        "surrogate_loss": surrogate_loss,
        "ratio": ratio,
        "log_ratio": log_ratio,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }


def gaussian_ppo_surrogate_loss(
    *,
    advantages: torch.Tensor,
    old_log_prob: torch.Tensor,
    new_log_prob: torch.Tensor,
    clip_range: float,
) -> dict[str, torch.Tensor]:
    if advantages.ndim == 1:
        advantages = advantages[:, None]
    if old_log_prob.ndim == 1:
        old_log_prob = old_log_prob[:, None]
    if new_log_prob.ndim == 1:
        new_log_prob = new_log_prob[:, None]
    if advantages.shape != old_log_prob.shape or advantages.shape != new_log_prob.shape:
        raise ValueError(
            "advantages, old_log_prob, and new_log_prob must have the same [B, 1] shape"
        )
    log_ratio = new_log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    surrogate_loss = torch.max(pg_loss1, pg_loss2).mean()
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
    return {
        "surrogate_loss": surrogate_loss,
        "ratio": ratio,
        "log_ratio": log_ratio,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }


def chunked_fpo_surrogate_loss(
    *,
    advantages: torch.Tensor,
    old_cfm_loss: torch.Tensor,
    new_cfm_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_range: float,
    cfm_diff_clip: float,
    trust_region_mode: str = "ppo",
    spo_clip_coef: float | None = None,
    cfm_loss_clamp: float = 0.0,
    log_ratio_scale: float = 1.0,
    cfm_diff_clip_min: float | None = None,
    cfm_diff_clip_max: float | None = None,
    clamp_negative_advantage_loss: float | None = None,
    positive_advantage_only: bool = False,
    average_cfm_loss_in_chunk: bool = False,
    average_cfm_loss_over_samples: bool = True,
) -> dict[str, torch.Tensor]:
    if old_cfm_loss.ndim != 3 or new_cfm_loss.ndim != 3:
        raise ValueError("old_cfm_loss and new_cfm_loss must be [B, T, S]")
    if old_cfm_loss.shape != new_cfm_loss.shape:
        raise ValueError("old_cfm_loss and new_cfm_loss must have the same shape")
    batch_size, chunk_steps, _ = old_cfm_loss.shape
    if valid_mask.shape != (batch_size, chunk_steps):
        raise ValueError(f"Expected valid_mask {(batch_size, chunk_steps)}, got {tuple(valid_mask.shape)}")
    if advantages.ndim == 2 and advantages.shape[1] == chunk_steps:
        valid_advantages = valid_mask.to(dtype=advantages.dtype)
        advantages = (advantages * valid_advantages).sum(dim=1, keepdim=True)
        advantages = advantages / valid_advantages.sum(dim=1, keepdim=True).clamp_min(1.0)
    elif advantages.ndim == 1:
        advantages = advantages[:, None]
    if advantages.shape != (batch_size, 1):
        raise ValueError(f"Expected advantages [B, 1] or [B, T], got {tuple(advantages.shape)}")

    if cfm_loss_clamp > 0:
        old_cfm_loss = clamp_ste(old_cfm_loss, max=cfm_loss_clamp)

    valid = valid_mask.to(dtype=old_cfm_loss.dtype).unsqueeze(-1)
    old_chunk = (old_cfm_loss * valid).sum(dim=1)
    new_chunk = (new_cfm_loss * valid).sum(dim=1)
    if average_cfm_loss_in_chunk:
        denom = valid.sum(dim=1).clamp_min(1.0)
        old_chunk = old_chunk / denom
        new_chunk = new_chunk / denom

    surrogate = fpo_surrogate_loss(
        advantages=advantages,
        old_cfm_loss=old_chunk,
        new_cfm_loss=new_chunk,
        clip_range=clip_range,
        cfm_diff_clip=cfm_diff_clip,
        trust_region_mode=trust_region_mode,
        spo_clip_coef=spo_clip_coef,
        cfm_loss_clamp=0.0,
        log_ratio_scale=log_ratio_scale,
        cfm_diff_clip_min=cfm_diff_clip_min,
        cfm_diff_clip_max=cfm_diff_clip_max,
        clamp_negative_advantage_loss=clamp_negative_advantage_loss,
        positive_advantage_only=positive_advantage_only,
        average_cfm_loss_over_samples=average_cfm_loss_over_samples,
    )
    surrogate["old_chunk_cfm_loss"] = old_chunk
    surrogate["new_chunk_cfm_loss"] = new_chunk
    return surrogate
