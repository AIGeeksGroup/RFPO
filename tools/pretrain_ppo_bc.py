#!/usr/bin/env python3
"""Pretrain an SB3 PPO actor from observation/action demonstrations."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _init_paths  # noqa: F401
from stable_baselines3 import PPO

from algos import ActorCriticPolicy
from hand_imitation.env.create_env import create_env
from hand_imitation.env.gym_wrapper import GymWrapper
from hand_imitation.utils.rm75_train_util import make_rm75_env


DEFAULT_SEQ = "ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"
DEFAULT_INFO_KEYWORDS = [
    "pregrasp_success",
    "pregrasp_steps",
    "imitate_steps",
    "hand_jpos_err",
    "obj_com_err",
    "obj_rot_err",
    "obj_lift",
    "contact_count",
    "stable_grasp_contact",
    "thumb_contact",
    "non_thumb_contact_count",
    "contact_success",
    "norm_success_3",
    "norm_success_10",
    "stage",
    "control_error",
    "obj_tgt_dist",
]


def _parse_dims(value: str, *, name: str) -> list[dict[str, list[int]]]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a Python-style list, got {value!r}") from exc
    if not isinstance(parsed, dict) or "pi" not in parsed or "vf" not in parsed:
        raise argparse.ArgumentTypeError(f"{name} must look like {{'pi':[...], 'vf':[...]}}")
    pi = [int(x) for x in parsed["pi"]]
    vf = [int(x) for x in parsed["vf"]]
    if not pi or not vf or any(x <= 0 for x in pi + vf):
        raise argparse.ArgumentTypeError(f"{name} must contain positive layer sizes")
    return [dict(pi=pi, vf=vf)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset", required=True, type=Path)
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument("--seq-name", default=DEFAULT_SEQ)
    parser.add_argument("--robot-name", default="rm75_inspire_right")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Optional exp_config.yaml to build a generic ViViDex env instead of the RM75 preset.",
    )
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss", choices=("huber", "mse", "l1", "nll"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--policy-net-arch", default="{'pi':[256,128], 'vf':[256,128]}")
    parser.add_argument("--log-std-init", type=float, default=-1.60)
    parser.add_argument("--target-log-std", type=float, default=-2.0)
    parser.add_argument("--object-scale", type=float, default=0.60)
    parser.add_argument("--no-rm75-native-profile", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    data = np.load(path)
    if "observations" not in data or "actions" not in data:
        raise ValueError(f"{path} must contain observations/actions arrays")
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations/actions must be rank-2 arrays")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations/actions row counts differ")
    finite = np.isfinite(observations).all(axis=1) & np.isfinite(actions).all(axis=1)
    observations = observations[finite]
    actions = actions[finite]
    if observations.shape[0] == 0:
        raise ValueError("dataset contains no finite samples")
    metadata = {
        "source": str(path),
        "samples": int(observations.shape[0]),
        "obs_dim": int(observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "kept_finite_fraction": float(finite.mean()),
    }
    for key in ("stage", "acceptance_rate", "attempts"):
        if key in data:
            value = np.asarray(data[key])
            metadata[key] = value.item() if value.shape == () else value.tolist()
    return observations, actions, metadata


def rm75_native_task_kwargs(object_scale: float) -> dict[str, object]:
    return {
        "action": "relocate",
        "object_scale": float(object_scale),
        "rm75_native_hand_control": True,
        "rm75_enforce_native_mimic_qpos": False,
        "rm75_pregrasp_success_mode": "proximity",
        "rm75_native_apply_template_approach_pregrasp": True,
        "rm75_pregrasp_palm_dist_thresh": 0.22,
        "rm75_pregrasp_min_finger_dist_thresh": 0.10,
        "rm75_done_on_pregrasp_failure": False,
        "rm75_default_approach_delta": [-0.0204, -0.1595, 0.1186],
        "rm75_override_template_approach_delta": True,
        "rm75_reset_settle_steps": 80,
        "rm75_force_imitate_steps": 260,
        "rm75_done_on_norm_success_10": True,
        "rm75_norm_success_lift_thresh": 0.08,
        "rm75_norm_success_requires_contact": True,
        "rm75_success_min_non_thumb_contacts": 2,
        "rm75_score_lift_target": 0.08,
        "rm75_ignore_pinky_contact": True,
        "rm75_disable_pinky_action": True,
        "rm75_pinky_action_value": -0.5,
        "rm75_scripted_action_prior": False,
        "rm75_scripted_action_prior_blend": 0.0,
        "reward_kwargs": {
            "pregrasp_success_thresh": 0.085,
            "obj_com_done_thresh": 0.35,
            "no_contact_grace_steps": 260,
            "controller_penalty_scale": 700.0,
            "action_penalty_scale": 0.008,
            "reward_divisor": 10.0,
            "finger_approach_reward_scale": 2.0,
            "finger_approach_scale": 22.0,
            "min_finger_approach_reward_scale": 2.0,
            "min_finger_approach_scale": 18.0,
            "palm_approach_reward_scale": 0.0,
            "palm_approach_scale": 7.0,
            "contact_reward_scale": 0.0,
            "thumb_contact_reward_scale": 6.0,
            "non_thumb_contact_reward_scale": 6.0,
            "required_non_thumb_contacts": 2,
            "object_reward_requires_contact": True,
            "object_reward_requires_stable_contact": True,
            "object_reward_contact_hold_steps": 2,
            "object_reward_scale": 9.0,
            "obj_err_scale": 45.0,
            "obj_rot_term": 0.1,
            "hand_mimic_reward_scale": 1.2,
            "hand_mimic_scale": 8.0,
            "hand_close_reward_scale": 0.80,
            "hand_open_penalty_scale": 2.20,
            "hand_close_target": 0.80,
            "reference_close_reward_scale": 0.25,
            "reference_close_penalty_scale": 1.20,
            "reference_close_tolerance": 0.04,
            "early_close_penalty_scale": 0.15,
            "hand_synergy_reward_scale": 0.40,
            "hand_synergy_penalty_scale": 0.20,
            "hand_synergy_min_main_close": 0.18,
            "stable_contact_bonus": 18.0,
            "contact_hold_bonus_scale": 0.0,
            "contact_hold_bonus_steps": 6,
            "lift_bonus_thresh": 0.012,
            "lift_bonus_mag": 8.0,
            "lift_reward_scale": 80.0,
            "lift_reward_cap": 0.08,
            "object_xy_drift_free_thresh": 0.018,
            "object_xy_drift_penalty_scale": 90.0,
            "object_speed_penalty_scale": 1.0,
            "object_tilt_penalty_scale": 10.0,
            "object_ang_vel_penalty_scale": 0.0,
            "unstable_push_penalty_scale": 35.0,
            "bad_push_done": False,
        },
    }


def _plain_config(value):
    if isinstance(value, dict):
        return {k: _plain_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain_config(v) for v in value]
    return value


def make_generic_env_from_config(
    config_file: Path,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
) -> tuple[PPO, object]:
    config = yaml.safe_load(config_file.read_text())
    params = config["params"] if "params" in config else config
    env_cfg = params["env"]
    task_kwargs = _plain_config(dict(env_cfg.get("task_kwargs", {})))
    base_env = create_env(
        name=env_cfg["name"],
        task_kwargs=task_kwargs,
        use_gui=False,
        is_eval=False,
        is_vision=False,
        norm_traj=bool(env_cfg.get("norm_traj", True)),
        robot_name=env_cfg.get("robot_name", "allegro_hand_ur5"),
    )
    env = GymWrapper(base_env)
    if env.observation_space.shape != (observations.shape[1],):
        env.close()
        raise ValueError(f"env obs shape {env.observation_space.shape} != dataset {(observations.shape[1],)}")
    if env.action_space.shape != (actions.shape[1],):
        env.close()
        raise ValueError(f"env action shape {env.action_space.shape} != dataset {(actions.shape[1],)}")

    agent_cfg = params.get("agent", {})
    policy_kwargs = agent_cfg.get("policy_kwargs")
    if not policy_kwargs:
        policy_kwargs = {
            "net_arch": [dict(pi=[256, 128], vf=[256, 128])],
            "log_std_init": -1.60,
        }
    model = PPO(
        ActorCriticPolicy,
        env,
        verbose=0,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        learning_rate=1e-5,
        policy_kwargs=_plain_config(policy_kwargs),
        seed=seed,
    )
    return model, env


def make_model(args: argparse.Namespace, observations: np.ndarray, actions: np.ndarray) -> tuple[PPO, object]:
    if args.config_file is not None:
        return make_generic_env_from_config(args.config_file, observations, actions, seed=args.seed)

    task_kwargs = {"action": "relocate"} if args.no_rm75_native_profile else rm75_native_task_kwargs(args.object_scale)
    env_kwargs = {
        "name": args.seq_name,
        "robot_name": args.robot_name,
        "norm_traj": True,
        "task_kwargs": task_kwargs,
        "info_keywords": DEFAULT_INFO_KEYWORDS,
        "n_envs": 1,
        "vid_freq": None,
        "vid_length": 100,
    }
    env = make_rm75_env(multi_proc=False, is_eval=False, **env_kwargs)
    if env.observation_space.shape != (observations.shape[1],):
        env.close()
        raise ValueError(f"env obs shape {env.observation_space.shape} != dataset {(observations.shape[1],)}")
    if env.action_space.shape != (actions.shape[1],):
        env.close()
        raise ValueError(f"env action shape {env.action_space.shape} != dataset {(actions.shape[1],)}")
    policy_kwargs = {
        "net_arch": _parse_dims(args.policy_net_arch, name="policy_net_arch"),
        "log_std_init": args.log_std_init,
    }
    model = PPO(
        ActorCriticPolicy,
        env,
        verbose=0,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        learning_rate=1e-5,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
    )
    return model, env


def action_mean(policy: ActorCriticPolicy, obs: torch.Tensor) -> torch.Tensor:
    latent_pi, _, _ = policy._get_latent(obs)
    return policy.action_net(latent_pi)


def bc_loss(policy: ActorCriticPolicy, obs: torch.Tensor, actions: torch.Tensor, mode: str, delta: float) -> torch.Tensor:
    pred = action_mean(policy, obs)
    if mode == "mse":
        return F.mse_loss(pred, actions)
    if mode == "l1":
        return F.l1_loss(pred, actions)
    if mode == "nll":
        _, log_prob, _ = policy.evaluate_actions(obs, actions)
        return -log_prob.mean()
    return F.huber_loss(pred, actions, reduction="mean", delta=delta)


@torch.no_grad()
def eval_bc_loss(
    policy: ActorCriticPolicy,
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
        losses.append(bc_loss(policy, observations[start:end], actions[start:end], mode, delta))
    if was_training:
        policy.train()
    return float(torch.stack(losses).mean().item())


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    observations_np, actions_np, metadata = load_dataset(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dataset_summary.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    model, env = make_model(args, observations_np, actions_np)
    device = resolve_device(args.device)
    model.policy.to(device)
    model.policy.train()
    if hasattr(model.policy, "log_std") and args.target_log_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(float(args.target_log_std))

    observations = torch.as_tensor(observations_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        list(model.policy.mlp_extractor.policy_net.parameters())
        + list(model.policy.action_net.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    n = observations.shape[0]
    batch_size = min(args.batch_size, n)
    best_loss = float("inf")
    print(
        "[pretrain-ppo-bc] "
        f"samples={n} obs_dim={observations.shape[1]} action_dim={actions.shape[1]} "
        f"steps={args.steps} device={device} output={args.output_dir}"
    )

    for step in range(1, args.steps + 1):
        batch_idx = torch.randint(0, n, (batch_size,), device=device)
        loss = bc_loss(
            model.policy,
            observations[batch_idx],
            actions[batch_idx],
            args.loss,
            args.huber_delta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), args.grad_clip)
        optimizer.step()

        should_log = step == 1 or step % args.log_freq == 0 or step == args.steps
        should_save = step % args.save_freq == 0 or step == args.steps
        if should_log or should_save:
            eval_loss = eval_bc_loss(
                model.policy,
                observations,
                actions,
                batch_size=max(batch_size * 8, batch_size),
                mode=args.loss,
                delta=args.huber_delta,
            )
            print(f"[pretrain-ppo-bc] step={step} train_loss={float(loss.item()):.6f} eval_loss={eval_loss:.6f}")
            if eval_loss < best_loss:
                best_loss = eval_loss
                model.save(args.output_dir / "bc_ppo_best")
            if should_save:
                model.save(args.output_dir / f"bc_ppo_step_{step}")

    model.save(args.output_dir / "bc_ppo_last")
    env.close()
    print(f"[pretrain-ppo-bc] saved {args.output_dir / 'bc_ppo_last.zip'} best_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
