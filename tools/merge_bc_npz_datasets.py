#!/usr/bin/env python3
"""Merge observation/action BC datasets stored as compressed NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("datasets", nargs="+", type=Path)
    return parser.parse_args()


def _as_array(data: np.lib.npyio.NpzFile, key: str, dtype) -> np.ndarray | None:
    if key not in data:
        return None
    return np.asarray(data[key], dtype=dtype)


def main() -> None:
    args = _parse_args()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    episode_ends: list[int] = []
    sources: list[dict[str, object]] = []
    offset = 0
    obs_dim: int | None = None
    action_dim: int | None = None

    for path in args.datasets:
        with np.load(path, allow_pickle=True) as data:
            obs = _as_array(data, "observations", np.float32)
            act = _as_array(data, "actions", np.float32)
            if obs is None or act is None:
                raise ValueError(f"{path} is missing observations/actions")
            if obs.ndim != 2 or act.ndim != 2 or obs.shape[0] != act.shape[0]:
                raise ValueError(f"{path} has invalid observation/action shapes: {obs.shape}, {act.shape}")
            if obs_dim is None:
                obs_dim = int(obs.shape[1])
                action_dim = int(act.shape[1])
            elif obs.shape[1] != obs_dim or act.shape[1] != action_dim:
                raise ValueError(
                    f"{path} dimensions ({obs.shape[1]}, {act.shape[1]}) do not match "
                    f"({obs_dim}, {action_dim})"
                )
            observations.append(obs)
            actions.append(act)
            reward = _as_array(data, "rewards", np.float32)
            rewards.append(reward if reward is not None and reward.shape[0] == obs.shape[0] else np.zeros(obs.shape[0], dtype=np.float32))
            ends = _as_array(data, "episode_ends", np.int64)
            if ends is None or ends.size == 0:
                episode_ends.append(offset + obs.shape[0])
            else:
                episode_ends.extend((ends + offset).astype(np.int64).tolist())
            sources.append(
                {
                    "path": str(path),
                    "samples": int(obs.shape[0]),
                    "episodes": int(ends.size if ends is not None else 1),
                    "source": str(np.asarray(data["source"]).item()) if "source" in data else "",
                }
            )
            offset += int(obs.shape[0])

    merged_obs = np.concatenate(observations, axis=0)
    merged_actions = np.concatenate(actions, axis=0)
    merged_rewards = np.concatenate(rewards, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=merged_obs,
        actions=merged_actions,
        rewards=merged_rewards,
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
        source=np.asarray("merged_bc_npz"),
        config_json=np.asarray(json.dumps({"sources": sources}, sort_keys=True)),
    )
    print(
        f"saved merged dataset to {args.output}; "
        f"samples={merged_obs.shape[0]} episodes={len(episode_ends)} "
        f"obs_dim={merged_obs.shape[1]} action_dim={merged_actions.shape[1]}"
    )


if __name__ == "__main__":
    main()
