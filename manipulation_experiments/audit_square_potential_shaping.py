#!/usr/bin/env python

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from src.potential_shaping import audit_potential_records, potential_shaping_term
from train_residual_flow_steering import (
    action_bounds,
    load_base_policy,
    make_environment,
    subset_observations,
)


@dataclass
class Config:
    base_policy_local_path: str = ""
    output_dir: str = "runs/square_potential_shaping_audit"
    device: str = "cuda"
    seed_start: int = 20260958
    num_episodes: int = 16
    num_envs: int = 16
    camera_size: int = 84
    sampling_steps: int = 10
    n_action_steps: int = 16
    discount: float = 0.995


def _make_block_config(cfg: Config, seed: int):
    return type(
        "BlockConfig",
        (),
        {
            "task": "Square",
            "num_envs": cfg.num_envs,
            "device": cfg.device,
            "camera_size": cfg.camera_size,
            "seed": seed,
        },
    )()


@torch.no_grad()
def collect(cfg: Config, actor) -> list[dict]:
    env = make_environment(_make_block_config(cfg, cfg.seed_start), actor)
    env.call("set_stage_potential_tracking", True)
    device = torch.device(cfg.device)
    low, high = action_bounds(env, device)
    observations, _ = env.reset()
    complete = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
    action_chunks: list[torch.Tensor | None] = [None] * cfg.num_envs
    action_offsets = [0] * cfg.num_envs
    episode_steps = [0] * cfg.num_envs
    records: list[dict] = []
    try:
        while not bool(complete.all()):
            replan_ids = torch.tensor(
                [
                    env_id
                    for env_id in range(cfg.num_envs)
                    if not complete[env_id] and action_chunks[env_id] is None
                ],
                device=device,
                dtype=torch.long,
            )
            if replan_ids.numel():
                sub_observations = subset_observations(observations, replan_ids)
                source = torch.randn(
                    replan_ids.numel(),
                    actor.config.horizon,
                    actor.model.action_dim,
                    device=device,
                )
                actions, _ = actor.predict_action_chunk(
                    sub_observations, source_noise=source
                )
                actions = actions[:, : cfg.n_action_steps].clamp(low, high)
                for local_id, env_tensor in enumerate(replan_ids):
                    env_id = int(env_tensor.item())
                    action_chunks[env_id] = actions[local_id]
                    action_offsets[env_id] = 0

            step_actions = []
            for env_id in range(cfg.num_envs):
                if complete[env_id]:
                    step_actions.append(torch.zeros_like(low))
                else:
                    chunk = action_chunks[env_id]
                    if chunk is None:
                        raise RuntimeError("active environment is missing an action chunk")
                    step_actions.append(chunk[action_offsets[env_id]])
            next_observations, _, terminated, truncated, _ = env.step(
                torch.stack(step_actions)
            )
            transitions = env.call("get_last_stage_transition")
            dones = terminated | truncated
            for env_id in range(cfg.num_envs):
                if complete[env_id]:
                    continue
                transition = transitions[env_id]
                term = potential_shaping_term(
                    transition["potential_before"],
                    transition["potential_next"],
                    discount=cfg.discount,
                    terminal=transition["terminal"],
                )
                records.append(
                    {
                        "seed": cfg.seed_start + env_id,
                        "step": episode_steps[env_id],
                        **transition,
                        "shaping_term": term,
                        "shaped_reward": transition["sparse_reward"] + term,
                    }
                )
                episode_steps[env_id] += 1
                action_offsets[env_id] += 1
                if bool(dones[env_id]):
                    complete[env_id] = True
                elif action_offsets[env_id] == cfg.n_action_steps:
                    action_chunks[env_id] = None
            observations = next_observations
    finally:
        env.close()
    return records


def main(cfg: Config) -> None:
    if not cfg.base_policy_local_path:
        raise ValueError("base_policy_local_path is required")
    if cfg.num_episodes != cfg.num_envs or cfg.num_envs <= 0:
        raise ValueError("the locked audit requires one episode per environment")
    random.seed(cfg.seed_start)
    np.random.seed(cfg.seed_start)
    torch.manual_seed(cfg.seed_start)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    actor = load_base_policy(
        cfg.base_policy_local_path,
        torch.device(cfg.device),
        cfg.sampling_steps,
        cfg.n_action_steps,
    )
    records = collect(cfg, actor)
    expected_seeds = set(range(cfg.seed_start, cfg.seed_start + cfg.num_episodes))
    results = {
        "config": asdict(cfg),
        **audit_potential_records(
            records, expected_seeds, discount=cfg.discount
        ),
    }
    with (output_dir / "records.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Config, config=(tyro.conf.FlagConversionOff,)))
