#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from src.parameter_space_es import (
    anchor_restored_exactly,
    batched_observation_hashes,
    snapshot_parameters,
)
from src.vine_returns import (
    array_sha256,
    candidate_rms,
    keyed_standard_normal,
    summarize_vine_records,
)
from train_residual_flow_steering import (
    action_bounds,
    load_base_policy,
    subset_observations,
)
from src.dexmg_env import create_vectorized_env


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_SEEDS = tuple(range(20261070, 20261078))
BRANCH_STEPS = (80, 160, 240)
CONTINUATION_REPLICAS = tuple(range(4))
NUM_CANDIDATES = 2
HORIZON_STEPS = 400
SAMPLING_STEPS = 10
ACTION_STEPS = 16
MAX_ENVS = 16


@dataclass
class Config:
    local_checkpoint_path: str
    output_dir: str
    mode: Literal["smoke", "audit"] = "smoke"
    device: str = "cuda"
    camera_size: int = 84


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _make_env(actor, seeds: list[int], cfg: Config):
    image_keys = [key.replace("observation.images.", "") for key in actor.config.image_features]
    return create_vectorized_env(
        env_name="Square",
        num_envs=len(seeds),
        device=cfg.device,
        camera_size=cfg.camera_size,
        video_key="agentview",
        expected_image_keys=image_keys,
        seeds=seeds,
    )


def _source(actor, key: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return keyed_standard_normal(
        (actor.config.horizon, actor.model.action_dim), *key
    ).to(device)


def _privileged_hashes(env) -> list[str]:
    return [array_sha256(np.asarray(value)) for value in env.call("get_privileged_state")]


@torch.no_grad()
def _predict_chunks(actor, observations, sources, low, high):
    actions, _ = actor.predict_action_chunk(observations, source_noise=sources)
    return actions[:, :ACTION_STEPS].clamp(low, high)


@torch.no_grad()
def collect_roots(actor, cfg: Config, seeds: tuple[int, ...], branch_steps: tuple[int, ...]):
    device = torch.device(cfg.device)
    env = _make_env(actor, list(seeds), cfg)
    low, high = action_bounds(env, device)
    observations, _ = env.reset()
    initial_hashes = batched_observation_hashes(observations)
    completed = np.zeros(len(seeds), dtype=bool)
    action_chunks: list[torch.Tensor | None] = [None] * len(seeds)
    action_offsets = [0] * len(seeds)
    replan_indices = [0] * len(seeds)
    prefix_actions: list[list[list[float]]] = [[] for _ in seeds]
    root_records: dict[int, dict[str, object]] = {
        seed: {
            "root_seed": seed,
            "initial_observation_hash": initial_hashes[index],
            "prefix_actions": prefix_actions[index],
            "branches": {},
        }
        for index, seed in enumerate(seeds)
    }
    max_step = max(branch_steps)
    try:
        for step in range(max_step):
            replan_ids = [
                index
                for index in range(len(seeds))
                if not completed[index] and action_chunks[index] is None
            ]
            if replan_ids:
                indices = torch.as_tensor(replan_ids, device=device, dtype=torch.long)
                sources = torch.stack(
                    [
                        _source(actor, (0, seeds[index], replan_indices[index]), device)
                        for index in replan_ids
                    ]
                )
                chunks = _predict_chunks(
                    actor,
                    subset_observations(observations, indices),
                    sources,
                    low,
                    high,
                )
                for local_index, env_index in enumerate(replan_ids):
                    action_chunks[env_index] = chunks[local_index]
                    action_offsets[env_index] = 0
                    replan_indices[env_index] += 1

            step_actions = []
            for env_index in range(len(seeds)):
                if completed[env_index]:
                    step_actions.append(torch.zeros_like(low))
                    continue
                chunk = action_chunks[env_index]
                if chunk is None:
                    raise RuntimeError("active root lane has no action chunk")
                action = chunk[action_offsets[env_index]]
                step_actions.append(action)
                prefix_actions[env_index].append(action.detach().cpu().tolist())
            observations, rewards, terminated, truncated, _ = env.step(torch.stack(step_actions))
            dones = (terminated | truncated).detach().cpu().numpy()
            for env_index in range(len(seeds)):
                if completed[env_index]:
                    continue
                if dones[env_index]:
                    completed[env_index] = True
                    continue
                action_offsets[env_index] += 1
                if action_offsets[env_index] == ACTION_STEPS:
                    action_chunks[env_index] = None

            reached_step = step + 1
            if reached_step not in branch_steps:
                continue
            observation_hashes = batched_observation_hashes(observations)
            privileged_hashes = _privileged_hashes(env)
            active_ids = [index for index in range(len(seeds)) if not completed[index]]
            for env_index in active_ids:
                seed = seeds[env_index]
                candidate_sources = torch.stack(
                    [
                        _source(actor, (1, seed, reached_step, candidate), device)
                        for candidate in range(NUM_CANDIDATES)
                    ]
                )
                repeated_observations = {
                    key: value[env_index : env_index + 1].expand(
                        NUM_CANDIDATES, *value.shape[1:]
                    )
                    for key, value in observations.items()
                }
                candidate_actions = _predict_chunks(
                    actor,
                    repeated_observations,
                    candidate_sources,
                    low,
                    high,
                )
                normalized_actions = actor.normalize_targets(
                    {"action": candidate_actions.clone()}
                )["action"]
                root_records[seed]["branches"][str(reached_step)] = {
                    "observation_hash": observation_hashes[env_index],
                    "privileged_state_hash": privileged_hashes[env_index],
                    "candidate_sources": candidate_sources.detach().cpu().tolist(),
                    "candidate_actions": candidate_actions.detach().cpu().tolist(),
                    "candidate_normalized_actions": normalized_actions.detach().cpu().tolist(),
                    "candidate_normalized_rms": candidate_rms(
                        normalized_actions[0].detach().cpu().numpy(),
                        normalized_actions[1].detach().cpu().numpy(),
                    ),
                }
    finally:
        env.close()
    return list(root_records.values())


def _scenario_batches(root_records, branch_steps, continuation_replicas):
    for branch_step in branch_steps:
        scenarios = []
        for root in root_records:
            if str(branch_step) not in root["branches"]:
                continue
            for candidate in range(NUM_CANDIDATES):
                for continuation in continuation_replicas:
                    scenarios.append(
                        {
                            "root_seed": int(root["root_seed"]),
                            "branch_step": branch_step,
                            "candidate": candidate,
                            "continuation_replica": continuation,
                        }
                    )
        for start in range(0, len(scenarios), MAX_ENVS):
            yield scenarios[start : start + MAX_ENVS]


@torch.no_grad()
def rollout_scenario_batch(actor, cfg: Config, roots_by_seed, scenarios):
    device = torch.device(cfg.device)
    seeds = [scenario["root_seed"] for scenario in scenarios]
    branch_step = scenarios[0]["branch_step"]
    if any(scenario["branch_step"] != branch_step for scenario in scenarios):
        raise ValueError("scenario batch must share one branch step")
    env = _make_env(actor, seeds, cfg)
    low, high = action_bounds(env, device)
    observations, _ = env.reset()
    initial_hashes = batched_observation_hashes(observations)
    prefix_done = np.zeros(len(scenarios), dtype=bool)
    try:
        for step in range(branch_step):
            actions = torch.as_tensor(
                [
                    roots_by_seed[scenario["root_seed"]]["prefix_actions"][step]
                    for scenario in scenarios
                ],
                device=device,
                dtype=torch.float32,
            )
            observations, _, terminated, truncated, _ = env.step(actions)
            prefix_done |= (terminated | truncated).detach().cpu().numpy()

        observation_hashes = batched_observation_hashes(observations)
        privileged_hashes = _privileged_hashes(env)
        replay_exact = []
        candidate_chunks = []
        for lane, scenario in enumerate(scenarios):
            root = roots_by_seed[scenario["root_seed"]]
            branch = root["branches"][str(branch_step)]
            exact = (
                not prefix_done[lane]
                and initial_hashes[lane] == root["initial_observation_hash"]
                and observation_hashes[lane] == branch["observation_hash"]
                and privileged_hashes[lane] == branch["privileged_state_hash"]
            )
            replay_exact.append(bool(exact))
            candidate_chunks.append(branch["candidate_actions"][scenario["candidate"]])

        completed = np.zeros(len(scenarios), dtype=bool)
        successes = np.zeros(len(scenarios), dtype=np.int64)
        episode_steps = np.full(len(scenarios), branch_step, dtype=np.int64)
        action_chunks = [
            torch.as_tensor(chunk, device=device, dtype=torch.float32)
            for chunk in candidate_chunks
        ]
        action_offsets = [0] * len(scenarios)
        continuation_replans = [0] * len(scenarios)

        while not completed.all():
            replan_ids = [
                index
                for index in range(len(scenarios))
                if not completed[index] and action_chunks[index] is None
            ]
            if replan_ids:
                indices = torch.as_tensor(replan_ids, device=device, dtype=torch.long)
                sources = torch.stack(
                    [
                        _source(
                            actor,
                            (
                                2,
                                scenarios[index]["root_seed"],
                                branch_step,
                                scenarios[index]["continuation_replica"],
                                continuation_replans[index],
                            ),
                            device,
                        )
                        for index in replan_ids
                    ]
                )
                chunks = _predict_chunks(
                    actor,
                    subset_observations(observations, indices),
                    sources,
                    low,
                    high,
                )
                for local_index, lane in enumerate(replan_ids):
                    action_chunks[lane] = chunks[local_index]
                    action_offsets[lane] = 0
                    continuation_replans[lane] += 1

            step_actions = []
            for lane in range(len(scenarios)):
                if completed[lane]:
                    step_actions.append(torch.zeros_like(low))
                    continue
                chunk = action_chunks[lane]
                if chunk is None:
                    raise RuntimeError("active vine lane has no action chunk")
                step_actions.append(chunk[action_offsets[lane]])
            observations, rewards, terminated, truncated, _ = env.step(torch.stack(step_actions))
            dones = (terminated | truncated).detach().cpu().numpy()
            reward_values = rewards.detach().cpu().numpy()
            for lane in range(len(scenarios)):
                if completed[lane]:
                    continue
                episode_steps[lane] += 1
                if dones[lane]:
                    completed[lane] = True
                    successes[lane] = int(reward_values[lane] == 1.0)
                else:
                    action_offsets[lane] += 1
                    if action_offsets[lane] == ACTION_STEPS:
                        action_chunks[lane] = None
                if episode_steps[lane] > HORIZON_STEPS:
                    raise RuntimeError("Square vine exceeded the locked 400-step horizon")

        return [
            {
                **scenario,
                "success": int(successes[lane]),
                "episode_length": int(episode_steps[lane]),
                "initial_observation_exact": initial_hashes[lane]
                == roots_by_seed[scenario["root_seed"]]["initial_observation_hash"],
                "branch_observation_exact": observation_hashes[lane]
                == roots_by_seed[scenario["root_seed"]]["branches"][str(branch_step)][
                    "observation_hash"
                ],
                "branch_privileged_state_exact": privileged_hashes[lane]
                == roots_by_seed[scenario["root_seed"]]["branches"][str(branch_step)][
                    "privileged_state_hash"
                ],
                "prefix_terminated_or_truncated": bool(prefix_done[lane]),
                "replay_exact": replay_exact[lane],
                "continuation_replans": continuation_replans[lane],
            }
            for lane, scenario in enumerate(scenarios)
        ]
    finally:
        env.close()


def main(cfg: Config) -> None:
    checkpoint = Path(cfg.local_checkpoint_path)
    if checkpoint.name != "trc7rbt0_step_110000":
        raise ValueError("H62 is locked to Square checkpoint trc7rbt0_step_110000")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(20261062)
    np.random.seed(20261062)
    torch.manual_seed(20261062)

    seeds = ROOT_SEEDS if cfg.mode == "audit" else ROOT_SEEDS[:1]
    branch_steps = BRANCH_STEPS if cfg.mode == "audit" else BRANCH_STEPS[:1]
    continuation_replicas = CONTINUATION_REPLICAS if cfg.mode == "audit" else (0,)
    actor = load_base_policy(
        str(checkpoint), torch.device(cfg.device), SAMPLING_STEPS, ACTION_STEPS
    )
    anchor = snapshot_parameters(actor)
    config_payload = {
        **asdict(cfg),
        "checkpoint_name": checkpoint.name,
        "root_seeds": list(seeds),
        "branch_steps": list(branch_steps),
        "num_candidates": NUM_CANDIDATES,
        "continuation_replicas": list(continuation_replicas),
        "sampling_steps": SAMPLING_STEPS,
        "action_steps": ACTION_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "renderer_claim": "OSMesa mechanism audit; not an official EGL benchmark result",
    }
    _atomic_json(output_dir / "config.json", config_payload)

    logger.info("Collecting %d root trajectories through step %d", len(seeds), max(branch_steps))
    root_records = collect_roots(actor, cfg, seeds, branch_steps)
    _atomic_json(output_dir / "roots.json", root_records)
    roots_by_seed = {int(record["root_seed"]): record for record in root_records}

    records = []
    for batch_index, scenarios in enumerate(
        _scenario_batches(root_records, branch_steps, continuation_replicas)
    ):
        logger.info("Running vine batch %d with %d lanes", batch_index, len(scenarios))
        records.extend(rollout_scenario_batch(actor, cfg, roots_by_seed, scenarios))
        _atomic_json(output_dir / "records.json", records)

    candidate_sources = []
    candidate_rms_values = []
    for root in root_records:
        for branch in root["branches"].values():
            candidate_sources.extend(branch["candidate_sources"])
            candidate_rms_values.append(branch["candidate_normalized_rms"])
    source_array = np.asarray(candidate_sources, dtype=np.float64)
    replay_exact_count = sum(bool(record["replay_exact"]) for record in records)
    validity = {
        "all_replays_exact": replay_exact_count == len(records),
        "replay_exact_count": replay_exact_count,
        "num_replays": len(records),
        "candidate_source_mean": float(source_array.mean()),
        "candidate_source_std": float(source_array.std()),
        "candidate_source_distribution": abs(float(source_array.mean())) <= 0.05
        and 0.95 <= float(source_array.std()) <= 1.05,
        "median_candidate_normalized_rms": float(np.median(candidate_rms_values)),
        "candidate_action_activity": float(np.median(candidate_rms_values)) >= 0.05,
        "policy_parameters_bitwise_unchanged": anchor_restored_exactly(actor, anchor),
        "all_outcomes_binary": all(int(record["success"]) in (0, 1) for record in records),
        "all_lengths_valid": all(
            int(record["branch_step"]) < int(record["episode_length"]) <= HORIZON_STEPS
            for record in records
        ),
    }
    validity["passed"] = all(
        value for key, value in validity.items() if isinstance(value, bool)
    )
    if cfg.mode == "audit":
        signal = summarize_vine_records(records)
        passed = bool(validity["passed"] and signal["passed"])
    else:
        signal = None
        passed = bool(validity["passed"] and len(records) == 2)
    results = {
        "hypothesis": "H62",
        "mode": cfg.mode,
        "num_root_records": len(root_records),
        "num_vine_states": sum(len(root["branches"]) for root in root_records),
        "num_suffix_rollouts": len(records),
        "validity": validity,
        "signal": signal,
        "passed": passed,
    }
    _atomic_json(output_dir / "results.json", results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Config, config=(tyro.conf.FlagConversionOff,)))
