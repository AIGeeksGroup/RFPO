#!/usr/bin/env python3
"""Pretrain an FPO state policy from observation/action demonstrations."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algos.rl.fpo_core import FPOPolicyConfig, FPOStatePolicy, sample_cfm_tensors
from algos.rl.fpo_trainer import save_fpo_checkpoint


def _parse_dims(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a Python-style list, got {value!r}") from exc
    if isinstance(parsed, int):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise argparse.ArgumentTypeError(f"{name} must be a non-empty list")
    dims = tuple(int(item) for item in parsed)
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return dims


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, type=Path)
    parser.add_argument("-o", "--output_dir", "--output-dir", dest="output_dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=0.0)
    parser.add_argument("--save_freq", "--save-freq", dest="save_freq", type=int, default=50000)
    parser.add_argument("--log_freq", "--log-freq", dest="log_freq", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--actor_hidden_dims", "--actor-hidden-dims", dest="actor_hidden_dims", default="[512,512]")
    parser.add_argument("--critic_hidden_dims", "--critic-hidden-dims", dest="critic_hidden_dims", default="[512,512]")
    parser.add_argument("--activation", default="elu")
    parser.add_argument("--timestep_embed_dim", "--timestep-embed-dim", dest="timestep_embed_dim", type=int, default=16)
    parser.add_argument("--sampling_steps", "--sampling-steps", dest="sampling_steps", type=int, default=8)
    parser.add_argument("--n_samples_per_action", "--n-samples-per-action", dest="n_samples_per_action", type=int, default=1)
    parser.add_argument("--actor_mlp_output_scale", "--actor-mlp-output-scale", dest="actor_mlp_output_scale", type=float, default=1.0)
    parser.add_argument("--actor_scale", "--actor-scale", dest="actor_scale", type=float, default=1.0)
    parser.add_argument("--action_clip", "--action-clip", dest="action_clip", type=float, default=1.0)
    parser.add_argument("--cfm_loss_t_inverse_cdf_beta", "--cfm-loss-t-inverse-cdf-beta", dest="cfm_loss_t_inverse_cdf_beta", type=float, default=1.5)
    parser.add_argument("--cfm_loss_reduction", "--cfm-loss-reduction", dest="cfm_loss_reduction", default="mean")
    parser.add_argument("--cfm_loss_use_huber", "--cfm-loss-use-huber", dest="cfm_loss_use_huber", action="store_true")
    parser.add_argument("--no_cfm_loss_use_huber", "--no-cfm-loss-use-huber", dest="cfm_loss_use_huber", action="store_false")
    parser.set_defaults(cfm_loss_use_huber=True)
    parser.add_argument("--cfm_loss_huber_delta", "--cfm-loss-huber-delta", dest="cfm_loss_huber_delta", type=float, default=1.0)
    parser.add_argument("--cfm_loss_huber_style", "--cfm-loss-huber-style", dest="cfm_loss_huber_style", default="torch")
    parser.add_argument("--flow_network_output_param", "--flow-network-output-param", dest="flow_network_output_param", default="u")
    parser.add_argument("--cfm_loss_mode", "--cfm-loss-mode", dest="cfm_loss_mode", default="u")
    parser.add_argument("--action_head_mode", "--action-head-mode", dest="action_head_mode", default="flow_residual")
    parser.add_argument("--residual_head_hidden_dims", "--residual-head-hidden-dims", dest="residual_head_hidden_dims", default="[128,64]")
    parser.add_argument("--residual_action_scale", "--residual-action-scale", dest="residual_action_scale", type=float, default=0.1)
    parser.add_argument("--direct_head_hidden_dims", "--direct-head-hidden-dims", dest="direct_head_hidden_dims", default="[256,128]")
    parser.add_argument("--direct_action_scale", "--direct-action-scale", dest="direct_action_scale", type=float, default=1.0)
    parser.add_argument("--action_loss_coef", "--action-loss-coef", dest="action_loss_coef", type=float, default=1.0)
    parser.add_argument("--cfm_loss_coef", "--cfm-loss-coef", dest="cfm_loss_coef", type=float, default=1.0)
    parser.add_argument("--action_loss", "--action-loss", dest="action_loss", choices=("huber", "mse", "l1"), default="huber")
    parser.add_argument("--action_huber_delta", "--action-huber-delta", dest="action_huber_delta", type=float, default=0.05)
    parser.add_argument("--grad_clip", "--grad-clip", dest="grad_clip", type=float, default=1.0)
    parser.add_argument("--obs_normalizer", "--obs-normalizer", dest="obs_normalizer", choices=("dataset", "identity"), default="dataset")
    parser.add_argument("--action_normalizer", "--action-normalizer", dest="action_normalizer", choices=("dataset", "identity"), default="dataset")
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    data = np.load(path)
    if "observations" not in data or "actions" not in data:
        raise ValueError(f"{path} must contain observations/actions arrays")
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations/actions must both be rank-2 arrays")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations/actions must have the same number of rows")
    finite = np.isfinite(observations).all(axis=1) & np.isfinite(actions).all(axis=1)
    observations = observations[finite]
    actions = actions[finite]
    if observations.shape[0] == 0:
        raise ValueError("dataset has no finite samples")
    metadata = {
        "source": str(path),
        "samples": int(observations.shape[0]),
        "obs_dim": int(observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "kept_finite_fraction": float(finite.mean()),
    }
    for key in ("episode_stable_steps", "episode_thumb_steps", "episode_non_thumb_steps", "episode_best_lift"):
        if key in data:
            arr = np.asarray(data[key], dtype=np.float32)
            metadata[key] = {
                "mean": float(arr.mean()) if arr.size else 0.0,
                "min": float(arr.min()) if arr.size else 0.0,
                "max": float(arr.max()) if arr.size else 0.0,
            }
    return observations, actions, metadata


def _normalizer(values: np.ndarray, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "identity":
        mean = np.zeros(values.shape[1], dtype=np.float32)
        std = np.ones(values.shape[1], dtype=np.float32)
    else:
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-3)
    return torch.from_numpy(mean), torch.from_numpy(std)


def _action_loss(predicted: torch.Tensor, target: torch.Tensor, mode: str, delta: float) -> torch.Tensor:
    if mode == "mse":
        return F.mse_loss(predicted, target)
    if mode == "l1":
        return F.l1_loss(predicted, target)
    return F.huber_loss(predicted, target, reduction="mean", delta=delta)


@torch.no_grad()
def _eval_action_loss(
    policy: FPOStatePolicy,
    observations: torch.Tensor,
    actions: torch.Tensor,
    *,
    batch_size: int,
    mode: str,
    delta: float,
) -> float:
    was_training = policy.training
    policy.eval()
    losses = []
    for start in range(0, observations.shape[0], batch_size):
        end = min(start + batch_size, observations.shape[0])
        pred = policy.act(observations[start:end], deterministic=True)
        losses.append(_action_loss(pred, actions[start:end], mode, delta).detach())
    if was_training:
        policy.train()
    return float(torch.stack(losses).mean().item())


def _save(
    *,
    path: Path,
    policy: FPOStatePolicy,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    metadata: dict[str, object],
) -> None:
    save_fpo_checkpoint(
        path=str(path),
        policy=policy,
        optimizer_state_dict=optimizer.state_dict(),
        num_timesteps=int(step),
        extra={
            "pretrain_fpo_bc_args": vars(args),
            "dataset": metadata,
        },
    )


def main() -> None:
    args = _parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    observations_np, actions_np, metadata = _load_dataset(args.dataset)
    device = _resolve_device(args.device)

    cfg = FPOPolicyConfig(
        obs_dim=observations_np.shape[1],
        action_dim=actions_np.shape[1],
        actor_hidden_dims=_parse_dims(args.actor_hidden_dims, name="actor_hidden_dims"),
        critic_hidden_dims=_parse_dims(args.critic_hidden_dims, name="critic_hidden_dims"),
        activation=args.activation,
        timestep_embed_dim=args.timestep_embed_dim,
        sampling_steps=args.sampling_steps,
        actor_mlp_output_scale=args.actor_mlp_output_scale,
        actor_scale=args.actor_scale,
        action_clip=args.action_clip,
        cfm_loss_t_inverse_cdf_beta=args.cfm_loss_t_inverse_cdf_beta,
        cfm_loss_reduction=args.cfm_loss_reduction,
        cfm_loss_use_huber=bool(args.cfm_loss_use_huber),
        cfm_loss_huber_delta=args.cfm_loss_huber_delta,
        cfm_loss_huber_style=args.cfm_loss_huber_style,
        flow_network_output_param=args.flow_network_output_param,
        cfm_loss_mode=args.cfm_loss_mode,
        action_head_mode=args.action_head_mode,
        residual_head_hidden_dims=_parse_dims(args.residual_head_hidden_dims, name="residual_head_hidden_dims"),
        residual_action_scale=args.residual_action_scale,
        direct_head_hidden_dims=_parse_dims(args.direct_head_hidden_dims, name="direct_head_hidden_dims"),
        direct_action_scale=args.direct_action_scale,
    )
    policy = FPOStatePolicy(cfg).to(device)

    obs_mean, obs_std = _normalizer(observations_np, args.obs_normalizer)
    action_mean, action_std = _normalizer(actions_np, args.action_normalizer)
    policy.set_normalizers(obs_mean, obs_std, action_mean, action_std)

    observations = torch.as_tensor(observations_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        list(policy.actor_parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dataset_summary.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    n = observations.shape[0]
    batch_size = min(args.batch_size, n)
    best_eval_loss = math.inf
    last_metrics: dict[str, float] = {}
    print(
        "[pretrain-fpo-bc] "
        f"samples={n} obs_dim={observations.shape[1]} action_dim={actions.shape[1]} "
        f"device={device} output={args.output_dir}"
    )

    for step in range(1, args.steps + 1):
        policy.train()
        batch_idx = torch.randint(0, n, (batch_size,), device=device)
        obs = observations[batch_idx]
        target_actions = actions[batch_idx]

        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        cfm_loss = torch.zeros((), dtype=torch.float32, device=device)
        if args.cfm_loss_coef > 0.0:
            eps, t = sample_cfm_tensors(
                batch_size=batch_size,
                n_samples=args.n_samples_per_action,
                action_dim=actions.shape[1],
                device=device,
                beta=policy.cfm_loss_t_inverse_cdf_beta,
            )
            per_sample_cfm, _, _ = policy.get_cfm_loss(obs, target_actions, eps, t)
            cfm_loss = per_sample_cfm.mean()
            total_loss = total_loss + float(args.cfm_loss_coef) * cfm_loss

        action_loss = torch.zeros((), dtype=torch.float32, device=device)
        if args.action_loss_coef > 0.0:
            predicted_actions = policy.act(obs, deterministic=True)
            action_loss = _action_loss(
                predicted_actions,
                target_actions,
                args.action_loss,
                args.action_huber_delta,
            )
            total_loss = total_loss + float(args.action_loss_coef) * action_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(policy.actor_parameters()), args.grad_clip)
        optimizer.step()

        if step % args.log_freq == 0 or step == 1 or step == args.steps:
            eval_batch = min(max(batch_size, 1), n)
            eval_idx = torch.randperm(n, device=device)[:eval_batch]
            eval_loss = _eval_action_loss(
                policy,
                observations[eval_idx],
                actions[eval_idx],
                batch_size=eval_batch,
                mode=args.action_loss,
                delta=args.action_huber_delta,
            )
            best_eval_loss = min(best_eval_loss, eval_loss)
            last_metrics = {
                "step": float(step),
                "loss": float(total_loss.detach().item()),
                "cfm_loss": float(cfm_loss.detach().item()),
                "action_loss": float(action_loss.detach().item()),
                "eval_action_loss": float(eval_loss),
                "best_eval_action_loss": float(best_eval_loss),
                "grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
            }
            print(
                "[pretrain-fpo-bc] "
                f"step={step} loss={last_metrics['loss']:.6f} "
                f"cfm={last_metrics['cfm_loss']:.6f} action={last_metrics['action_loss']:.6f} "
                f"eval_action={eval_loss:.6f} best={best_eval_loss:.6f} "
                f"grad={last_metrics['grad_norm']:.3f}"
            )

        if args.save_freq > 0 and step % args.save_freq == 0:
            _save(
                path=args.output_dir / f"flow_bc_step_{step}.pt",
                policy=policy,
                optimizer=optimizer,
                step=step,
                args=args,
                metadata=metadata,
            )

    _save(
        path=args.output_dir / "flow_bc_last.pt",
        policy=policy,
        optimizer=optimizer,
        step=args.steps,
        args=args,
        metadata=metadata,
    )
    with (args.output_dir / "metrics_last.json").open("w") as f:
        json.dump(last_metrics, f, indent=2, sort_keys=True)
    print(f"[pretrain-fpo-bc] saved {args.output_dir / 'flow_bc_last.pt'}")


if __name__ == "__main__":
    main()
