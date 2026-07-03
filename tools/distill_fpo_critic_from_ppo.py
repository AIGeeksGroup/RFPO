#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import _init_paths

from algos.rl.fpo_core import FPOPolicyConfig, FPOStatePolicy
from algos.rl.fpo_trainer import save_fpo_checkpoint


def load_fpo_checkpoint(path: str, device: torch.device) -> tuple[FPOStatePolicy, dict]:
    checkpoint = torch.load(path, map_location=device)
    policy = FPOStatePolicy(FPOPolicyConfig.from_mapping(checkpoint["policy_config"])).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    normalizers = checkpoint.get("normalizers")
    if normalizers:
        policy.set_normalizers(
            normalizers["obs_mean"],
            normalizers["obs_std"],
            normalizers["action_mean"],
            normalizers["action_std"],
        )
    return policy, checkpoint


def ppo_predict_values(ppo_model, observations: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    values: list[np.ndarray] = []
    ppo_model.policy.to(device)
    ppo_model.policy.eval()
    for start in range(0, len(observations), batch_size):
        obs = torch.as_tensor(observations[start : start + batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            try:
                pred = ppo_model.policy.predict_values(obs)
            except AttributeError:
                _actions, pred, _log_prob = ppo_model.policy(obs, deterministic=True)
        values.append(pred.detach().cpu().numpy().reshape(-1, 1))
    return np.concatenate(values, axis=0).astype(np.float32)


def distill_critic(
    *,
    policy: FPOStatePolicy,
    observations: np.ndarray,
    target_values: np.ndarray,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    device: torch.device,
    log_freq: int,
) -> dict[str, float]:
    for param in policy.actor.parameters():
        param.requires_grad_(False)
    for param in policy.critic.parameters():
        param.requires_grad_(True)
    policy.train()

    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(target_values, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        policy.critic.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-5,
    )
    n = obs_tensor.shape[0]

    with torch.no_grad():
        initial_loss = F.mse_loss(policy.evaluate(obs_tensor), target_tensor).item()

    loss_value = initial_loss
    for step in range(1, steps + 1):
        batch_idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        pred = policy.evaluate(obs_tensor[batch_idx])
        loss = F.mse_loss(pred, target_tensor[batch_idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), max_grad_norm)
        optimizer.step()
        loss_value = float(loss.item())
        if step == 1 or (log_freq > 0 and step % log_freq == 0):
            print(f"step={step} critic_distill_loss={loss_value:.6f}", flush=True)

    policy.eval()
    with torch.no_grad():
        final_loss = F.mse_loss(policy.evaluate(obs_tensor), target_tensor).item()
    return {"initial_mse": float(initial_loss), "final_mse": float(final_loss), "last_batch_mse": loss_value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpo_checkpoint", required=True)
    parser.add_argument("--ppo_checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--value_batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_freq", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    data = np.load(args.dataset)
    observations = np.asarray(data["observations"], dtype=np.float32)
    if observations.ndim != 2:
        raise ValueError("Dataset observations must be a rank-2 array")

    from stable_baselines3 import PPO

    policy, checkpoint = load_fpo_checkpoint(args.fpo_checkpoint, device)
    ppo_model = PPO.load(args.ppo_checkpoint, device=device)
    target_values = ppo_predict_values(ppo_model, observations, device, args.value_batch_size)

    stats = distill_critic(
        policy=policy,
        observations=observations,
        target_values=target_values,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        device=device,
        log_freq=args.log_freq,
    )
    extra = dict(checkpoint.get("extra", {}))
    extra.update(
        {
            "critic_distilled_from": args.ppo_checkpoint,
            "critic_distill_dataset": args.dataset,
            "critic_distill_steps": args.steps,
            "critic_distill_initial_mse": stats["initial_mse"],
            "critic_distill_final_mse": stats["final_mse"],
        }
    )
    save_fpo_checkpoint(
        path=args.output,
        policy=policy,
        optimizer_state_dict=None,
        num_timesteps=int(checkpoint.get("num_timesteps", 0)),
        extra=extra,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(
        "saved value-distilled checkpoint to "
        f"{args.output} initial_mse={stats['initial_mse']:.6f} final_mse={stats['final_mse']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
