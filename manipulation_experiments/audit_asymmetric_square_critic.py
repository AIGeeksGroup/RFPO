#!/usr/bin/env python

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import tyro

from src.asymmetric_critic import (
    discounted_success_labels,
    fit_and_evaluate,
    stratified_episode_folds,
    validate_collection,
)
from train_residual_flow_steering import (
    action_bounds,
    encode,
    load_base_policy,
    make_environment,
    subset_observations,
)


@dataclass
class Config:
    base_policy_local_path: str = ""
    output_dir: str = "runs/asymmetric_square_critic"
    device: str = "cuda"
    seed_start: int = 20260926
    num_episodes: int = 32
    num_envs: int = 16
    camera_size: int = 84
    sampling_steps: int = 10
    n_action_steps: int = 16
    discount: float = 0.995
    critic_epochs: int = 10
    critic_minibatches: int = 8
    critic_learning_rate: float = 1e-4


def _empty_plans(num_envs: int) -> dict:
    return {
        "active": torch.zeros(num_envs, dtype=torch.bool),
        "actions": [None] * num_envs,
        "duration": [0] * num_envs,
        "record": [None] * num_envs,
    }


@torch.no_grad()
def collect_block(cfg: Config, actor, seeds: list[int]) -> list[dict]:
    block_cfg = type("BlockConfig", (), {
        "task": "Square",
        "num_envs": len(seeds),
        "device": cfg.device,
        "camera_size": cfg.camera_size,
        "seed": seeds[0],
    })()
    env = make_environment(block_cfg, actor)
    # make_environment uses consecutive seed + env_id, exactly matching this block.
    if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
        raise ValueError("collection blocks require consecutive seeds")
    device = torch.device(cfg.device)
    low, high = action_bounds(env, device)
    observations, _ = env.reset()
    plans = _empty_plans(len(seeds))
    complete = torch.zeros(len(seeds), dtype=torch.bool, device=device)
    records: list[dict] = []
    try:
        while not bool(complete.all()):
            start_ids = torch.where(~complete.cpu() & ~plans["active"])[0].to(device)
            if start_ids.numel():
                sub_observations = subset_observations(observations, start_ids)
                conditioning = encode(actor, sub_observations)
                source = torch.randn(
                    start_ids.numel(),
                    actor.config.horizon,
                    actor.model.action_dim,
                    device=device,
                )
                actions, _ = actor.predict_action_chunk(sub_observations, source_noise=source)
                actions = actions.clamp(low, high)
                privileged_all = env.call("get_privileged_state")
                for local_id, env_tensor in enumerate(start_ids):
                    env_id = int(env_tensor.item())
                    plans["active"][env_id] = True
                    plans["actions"][env_id] = actions[local_id]
                    plans["duration"][env_id] = 0
                    plans["record"][env_id] = {
                        "seed": seeds[env_id],
                        "visual": conditioning[local_id].cpu(),
                        "privileged": torch.as_tensor(
                            privileged_all[env_id], dtype=torch.float32
                        ),
                        "reward": 0.0,
                        "duration": 0,
                        "terminal": False,
                        "success": False,
                    }

            step_actions = []
            for env_id in range(len(seeds)):
                if complete[env_id]:
                    step_actions.append(torch.zeros_like(low))
                else:
                    step_actions.append(
                        plans["actions"][env_id][plans["duration"][env_id]]
                    )
            next_observations, rewards, terminated, truncated, _ = env.step(
                torch.stack(step_actions)
            )
            dones = terminated | truncated
            for env_id in range(len(seeds)):
                if complete[env_id]:
                    continue
                duration = plans["duration"][env_id]
                record = plans["record"][env_id]
                record["reward"] += cfg.discount**duration * float(rewards[env_id].item())
                plans["duration"][env_id] += 1
                if bool(dones[env_id]) or plans["duration"][env_id] == cfg.n_action_steps:
                    record["duration"] = plans["duration"][env_id]
                    record["terminal"] = bool(dones[env_id])
                    record["success"] = bool(
                        dones[env_id] and rewards[env_id].item() == 1.0
                    )
                    records.append(record)
                    plans["active"][env_id] = False
                if bool(dones[env_id]):
                    complete[env_id] = True
            observations = next_observations
    finally:
        env.close()
    return records


def add_labels(records: list[dict], discount: float) -> None:
    for seed in sorted({int(record["seed"]) for record in records}):
        episode = [record for record in records if int(record["seed"]) == seed]
        labels = discounted_success_labels(
            [int(record["duration"]) for record in episode],
            [float(record["reward"]) for record in episode],
            discount,
        )
        for record, label in zip(episode, labels, strict=True):
            record["label"] = float(label.item())


def stack_fold(records: list[dict], seeds: set[int], key: str, device: torch.device):
    selected = [record for record in records if int(record["seed"]) in seeds]
    inputs = torch.stack([torch.as_tensor(record[key]) for record in selected]).to(device)
    labels = torch.tensor([record["label"] for record in selected], device=device)
    return inputs, labels


def run_cross_fit(cfg: Config, records: list[dict], outcomes: dict[int, bool]) -> dict:
    folds = stratified_episode_folds(outcomes)
    device = torch.device(cfg.device)
    fold_results = []
    for evaluation_fold in range(2):
        train_seeds = folds[1 - evaluation_fold]
        evaluation_seeds = folds[evaluation_fold]
        result = {
            "fold": evaluation_fold,
            "train_seeds": sorted(train_seeds),
            "evaluation_seeds": sorted(evaluation_seeds),
        }
        for model_id, key in enumerate(("visual", "privileged")):
            train_inputs, train_labels = stack_fold(records, train_seeds, key, device)
            evaluation_inputs, evaluation_labels = stack_fold(
                records, evaluation_seeds, key, device
            )
            result[key] = fit_and_evaluate(
                train_inputs,
                train_labels,
                evaluation_inputs,
                evaluation_labels,
                seed=cfg.seed_start + 100 * evaluation_fold + model_id,
                epochs=cfg.critic_epochs,
                num_minibatches=cfg.critic_minibatches,
                learning_rate=cfg.critic_learning_rate,
            )
        result["mse_relative_change"] = (
            result["privileged"]["mse"] / result["visual"]["mse"] - 1.0
        )
        result["spearman_gain"] = (
            result["privileged"]["spearman"] - result["visual"]["spearman"]
        )
        fold_results.append(result)
    passed = all(
        fold["mse_relative_change"] <= -0.10
        and fold["spearman_gain"] >= 0.10
        and fold["privileged"]["spearman"] > 0.0
        for fold in fold_results
    )
    return {"folds": fold_results, "passed": passed}


def main(cfg: Config) -> None:
    if not cfg.base_policy_local_path:
        raise ValueError("base_policy_local_path is required")
    if cfg.num_episodes <= 0 or cfg.num_episodes % cfg.num_envs:
        raise ValueError("num_episodes must be a positive multiple of num_envs")
    random.seed(cfg.seed_start)
    np.random.seed(cfg.seed_start)
    torch.manual_seed(cfg.seed_start)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    device = torch.device(cfg.device)
    actor = load_base_policy(
        cfg.base_policy_local_path, device, cfg.sampling_steps, cfg.n_action_steps
    )
    records: list[dict] = []
    for start in range(cfg.seed_start, cfg.seed_start + cfg.num_episodes, cfg.num_envs):
        seeds = list(range(start, start + cfg.num_envs))
        records.extend(collect_block(cfg, actor, seeds))
    expected_seeds = set(range(cfg.seed_start, cfg.seed_start + cfg.num_episodes))
    outcomes = validate_collection(records, expected_seeds)
    add_labels(records, cfg.discount)
    torch.save(records, output_dir / "records.pt")
    results = {
        "config": asdict(cfg),
        "episodes": len(outcomes),
        "successes": sum(outcomes.values()),
        "failures": len(outcomes) - sum(outcomes.values()),
        "plans": len(records),
        **run_cross_fit(cfg, records, outcomes),
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Config, config=(tyro.conf.FlagConversionOff,)))
