#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

import imageio
import numpy as np
import yaml

import _init_paths  # noqa: F401
from algos.rl.fpo_trainer import load_fpo_state_policy
from hand_imitation.env.create_env import create_env
from hand_imitation.env.gym_wrapper import GymWrapper
from stable_baselines3 import PPO


def infer_run_dir(checkpoint: str) -> Path:
    p = Path(checkpoint)
    if p.name in {"best.pt", "last.pt", "restore_checkpoint.pt"}:
        return p.parents[1]
    if p.suffix == ".pt" and p.parent.name == "models":
        return p.parents[1]
    if p.suffix == ".zip" and p.parent.name == "models":
        return p.parents[1]
    if p.name == "restore_checkpoint.zip":
        return p.parent
    return p


def _expected_obs_dim(policy):
    try:
        return int(policy.policy.obs_mean.shape[-1])
    except Exception:
        pass
    try:
        return int(policy.observation_space.shape[-1])
    except Exception:
        return None


def infer_algo_name(config, checkpoint: str) -> str:
    agent_name = config.get("params", {}).get("agent", {}).get("name", "PPO")
    if agent_name == "FPO" or str(checkpoint).endswith(".pt"):
        return "fpo"
    return "ppo"


def load_policy(config, checkpoint: str):
    algo_name = infer_algo_name(config, checkpoint)
    if algo_name == "fpo":
        return load_fpo_state_policy(checkpoint)
    return PPO.load(checkpoint)


def set_nested_override(target: dict, key: str, value):
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError(f"Empty override key from {key!r}")
    current = target
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set nested override {key!r}: {part!r} is not a dict")
        current = child
    current[parts[-1]] = value


def _adapt_obs(policy, obs):
    if isinstance(obs, dict):
        return obs
    obs = np.asarray(obs, dtype=np.float32)
    expected = _expected_obs_dim(policy)
    if expected is None or obs.shape[-1] == expected:
        return obs
    fixed = np.zeros((expected,), dtype=np.float32)
    copy = min(int(obs.shape[-1]), expected)
    fixed[:copy] = obs[:copy]
    return fixed


def predict(policy, obs, deterministic=True):
    obs = _adapt_obs(policy, obs)
    batched = obs[None, :] if not isinstance(obs, dict) else {k: v[None, :] for k, v in obs.items()}
    try:
        action = policy.predict(observation=batched, deterministic=deterministic)[0]
    except Exception:
        fix_obs = np.zeros((1, 396), dtype=np.float32)
        fix_obs[:, :367] = batched[:, :367]
        fix_obs[:, 367:370] = batched[:, 364:367]
        fix_obs[:, 370:] = batched[:, 367:]
        action = policy.predict(observation=fix_obs, deterministic=deterministic)[0]
    if len(action.shape) > 1:
        action = action[0]
    return action


def _jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def summarize_trace(trace, final_info, total_reward, frames, sr3, sr10):
    def values(key):
        out = []
        for info in trace:
            value = info.get(key)
            if isinstance(value, (bool, np.bool_)):
                out.append(float(value))
            elif isinstance(value, (int, float, np.integer, np.floating)):
                out.append(float(value))
        return out

    def max_value(key, default=0.0):
        vals = values(key)
        return max(vals) if vals else default

    def final_value(key, default=0.0):
        value = final_info.get(key, default)
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        return default

    stable_flags = values("stable_grasp_contact")
    contact_counts = values("contact_count")
    obj_com_errs = values("obj_com_err")
    hand_jpos_errs = values("hand_jpos_err")
    hand_mjpos_errs = values("hand_mjpos_err")
    drift_xs = values("object_xy_drift_x")
    drift_ys = values("object_xy_drift_y")
    return {
        "total_reward": float(total_reward),
        "frames": int(frames),
        "sr3": float(sr3),
        "sr10": float(sr10),
        "final_stage": final_value("stage"),
        "final_pregrasp_success": final_value("pregrasp_success"),
        "final_obj_com_err": final_value("obj_com_err"),
        "min_obj_com_err": min(obj_com_errs) if obj_com_errs else 0.0,
        "final_hand_jpos_err": final_value("hand_jpos_err"),
        "min_hand_jpos_err": min(hand_jpos_errs) if hand_jpos_errs else 0.0,
        "final_hand_mjpos_err": final_value("hand_mjpos_err"),
        "min_hand_mjpos_err": min(hand_mjpos_errs) if hand_mjpos_errs else 0.0,
        "final_obj_lift": final_value("obj_lift"),
        "max_obj_lift": max_value("obj_lift"),
        "final_contact_count": final_value("contact_count"),
        "max_contact_count": max(contact_counts) if contact_counts else 0.0,
        "contact_frames": int(sum(1 for v in contact_counts if v > 0.0)),
        "stable_contact_frames": int(sum(1 for v in stable_flags if v > 0.0)),
        "final_stable_contact_hold_steps": final_value("stable_contact_hold_steps"),
        "max_stable_contact_hold_steps": max_value("stable_contact_hold_steps"),
        "final_object_xy_drift": final_value("object_xy_drift"),
        "max_object_xy_drift": max_value("object_xy_drift"),
        "final_object_xy_drift_x": final_value("object_xy_drift_x"),
        "final_object_xy_drift_y": final_value("object_xy_drift_y"),
        "max_abs_object_xy_drift_x": max((abs(v) for v in drift_xs), default=0.0),
        "max_abs_object_xy_drift_y": max((abs(v) for v in drift_ys), default=0.0),
        "final_object_tilt_err": final_value("object_tilt_err"),
        "max_object_tilt_err": max_value("object_tilt_err"),
        "final_rm75_grasp_score": final_value("rm75_grasp_score"),
        "max_rm75_grasp_score": max_value("rm75_grasp_score"),
        "final_rm75_task_score": final_value("rm75_task_score"),
        "max_rm75_task_score": max_value("rm75_task_score"),
        "final_contact_success": final_value("contact_success"),
        "max_contact_success": max_value("contact_success"),
        "final_lift_success_5cm": final_value("lift_success_5cm"),
        "max_lift_success_5cm": max_value("lift_success_5cm"),
        "final_norm_success_10": final_value("norm_success_10"),
        "max_norm_success_10": max_value("norm_success_10"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-render", action="store_true", help="Run rollout metrics without RGB rendering or mp4 output.")
    parser.add_argument(
        "--task-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a task_kwargs entry for diagnostic rollouts. VALUE is parsed as YAML.",
    )
    args = parser.parse_args()

    if args.no_render:
        os.environ["VIVIDEX_HEADLESS_NO_RENDER"] = "1"
    else:
        os.environ.pop("VIVIDEX_HEADLESS_NO_RENDER", None)
    if args.seed is not None:
        np.random.seed(args.seed)
    run_dir = infer_run_dir(args.checkpoint)
    config = yaml.safe_load(open(run_dir / "exp_config.yaml", "r"))
    env_cfg = config["params"]["env"]
    task_kwargs = dict(env_cfg["task_kwargs"])
    for item in args.task_override:
        if "=" not in item:
            raise ValueError(f"--task-override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        set_nested_override(task_kwargs, key, yaml.safe_load(value))
    policy = load_policy(config, args.checkpoint)

    base_env = create_env(
        name=env_cfg["name"],
        task_kwargs=task_kwargs,
        use_gui=False,
        is_eval=True,
        is_vision=False,
        norm_traj=bool(env_cfg.get("norm_traj", True)),
        robot_name=env_cfg.get("robot_name", "allegro_hand_ur5"),
    )
    base_env._stage = args.stage
    env = GymWrapper(base_env)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    csv_rows = []
    for ep in range(args.episodes):
        obs = env.reset()
        frames = [] if args.no_render else [env.render(mode="rgb_array")]
        total_reward = 0.0
        done = False
        info = {}
        trace = []
        for step in range(min(args.max_steps, getattr(base_env, "horizon", args.max_steps))):
            action = predict(policy, obs, deterministic=not args.stochastic)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            trace.append({key: _jsonable(value) for key, value in info.items()})
            if not args.no_render:
                frames.append(env.render(mode="rgb_array"))
            if done:
                break
        sr3, sr10 = env.is_success()
        stem = f"rfpo_rollout_stage{args.stage}_ep{ep:02d}_reward{total_reward:.1f}"
        path = out_dir / f"{stem}.mp4"
        if not args.no_render:
            writer = imageio.get_writer(path, fps=args.fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        metrics = summarize_trace(trace, info, total_reward, len(frames), sr3, sr10)
        output_name = path.name if not args.no_render else f"{stem}_trace.json"
        summary.append((output_name, metrics, {key: _jsonable(value) for key, value in info.items()}))
        csv_rows.append({"video": output_name, **metrics})
        with open(out_dir / f"{stem}_trace.json", "w") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        print(
            f"saved {output_name} reward={total_reward:.3f} frames={len(frames)} "
            f"sr3={float(sr3)} sr10={float(sr10)} "
            f"max_lift={metrics['max_obj_lift']:.4f} "
            f"stable_frames={metrics['stable_contact_frames']} "
            f"max_task={metrics['max_rm75_task_score']:.2f}"
        )
    with open(out_dir / "summary.txt", "w") as f:
        for row in summary:
            f.write(repr(row) + "\n")
    if csv_rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
