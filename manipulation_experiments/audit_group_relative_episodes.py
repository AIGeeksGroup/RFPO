#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from src.dexmg_env import create_vectorized_env
from src.group_relative_returns import summarize_group_relative_audit
from src.parameter_space_es import (
    anchor_restored_exactly,
    batched_observation_hashes,
    snapshot_parameters,
)
from src.vine_returns import array_sha256, candidate_rms, keyed_standard_normal
from train_residual_flow_steering import (
    action_bounds,
    load_base_policy,
    subset_observations,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT_SEEDS = tuple(range(20261080, 20261088))
FORMAL_REPLICAS = 4
SMOKE_REPLICAS = 2
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


def _source(actor, root_seed: int, replica: int, plan_index: int) -> torch.Tensor:
    return keyed_standard_normal(
        (actor.config.horizon, actor.model.action_dim),
        63,
        root_seed,
        replica,
        plan_index,
    )


@torch.no_grad()
def _rollout_batch(actor, cfg: Config, scenarios: list[tuple[int, int]]):
    device = torch.device(cfg.device)
    seeds = [root_seed for root_seed, _ in scenarios]
    env = _make_env(actor, seeds, cfg)
    low, high = action_bounds(env, device)
    observations, _ = env.reset()
    initial_hashes = batched_observation_hashes(observations)
    privileged_hashes = [
        array_sha256(np.asarray(value)) for value in env.call("get_privileged_state")
    ]
    completed = np.zeros(len(scenarios), dtype=bool)
    successes = np.zeros(len(scenarios), dtype=np.int64)
    lengths = np.zeros(len(scenarios), dtype=np.int64)
    chunks: list[torch.Tensor | None] = [None] * len(scenarios)
    offsets = [0] * len(scenarios)
    plan_indices = [0] * len(scenarios)
    first_normalized_chunks: list[list[list[float]] | None] = [None] * len(scenarios)
    source_sum = 0.0
    source_square_sum = 0.0
    source_count = 0

    try:
        for _ in range(HORIZON_STEPS):
            replan_lanes = [
                lane
                for lane in range(len(scenarios))
                if not completed[lane] and chunks[lane] is None
            ]
            if replan_lanes:
                indices = torch.as_tensor(replan_lanes, device=device, dtype=torch.long)
                cpu_sources = torch.stack(
                    [
                        _source(actor, *scenarios[lane], plan_indices[lane])
                        for lane in replan_lanes
                    ]
                )
                source_sum += float(cpu_sources.double().sum().item())
                source_square_sum += float(cpu_sources.double().square().sum().item())
                source_count += cpu_sources.numel()
                predicted, _ = actor.predict_action_chunk(
                    subset_observations(observations, indices),
                    source_noise=cpu_sources.to(device),
                )
                predicted = predicted[:, :ACTION_STEPS].clamp(low, high)
                normalized = actor.normalize_targets({"action": predicted.clone()})["action"]
                for local_index, lane in enumerate(replan_lanes):
                    chunks[lane] = predicted[local_index]
                    offsets[lane] = 0
                    plan_indices[lane] += 1
                    if first_normalized_chunks[lane] is None:
                        first_normalized_chunks[lane] = normalized[local_index].cpu().tolist()

            actions = []
            for lane in range(len(scenarios)):
                if completed[lane]:
                    actions.append(torch.zeros_like(low))
                    continue
                chunk = chunks[lane]
                if chunk is None:
                    raise RuntimeError("active episode lane has no action chunk")
                actions.append(chunk[offsets[lane]])

            observations, rewards, terminated, truncated, _ = env.step(torch.stack(actions))
            reward_values = rewards.detach().cpu().numpy()
            if not np.isfinite(reward_values).all():
                raise FloatingPointError("group-relative rollout produced non-finite rewards")
            active = ~completed
            lengths[active] += 1
            done_values = (terminated | truncated).detach().cpu().numpy()
            newly_done = active & done_values
            successes[newly_done] = (reward_values[newly_done] == 1.0).astype(np.int64)
            completed[newly_done] = True

            for lane in range(len(scenarios)):
                if completed[lane]:
                    continue
                offsets[lane] += 1
                if offsets[lane] == ACTION_STEPS:
                    chunks[lane] = None

            if completed.all():
                break
    finally:
        env.close()

    records = []
    for lane, (root_seed, replica) in enumerate(scenarios):
        first_chunk = first_normalized_chunks[lane]
        if first_chunk is None:
            raise RuntimeError("episode lane produced no action chunk")
        records.append(
            {
                "root_seed": root_seed,
                "replica": replica,
                "success": int(successes[lane]),
                "completed": bool(completed[lane]),
                "length": int(lengths[lane]),
                "plan_count": int(plan_indices[lane]),
                "initial_observation_hash": initial_hashes[lane],
                "initial_privileged_state_hash": privileged_hashes[lane],
                "first_normalized_action_chunk": first_chunk,
            }
        )
    return records, source_sum, source_square_sum, source_count


def _scenario_batches(root_seeds: tuple[int, ...], replicas: int):
    scenarios = [(root_seed, replica) for root_seed in root_seeds for replica in range(replicas)]
    for start in range(0, len(scenarios), MAX_ENVS):
        yield scenarios[start : start + MAX_ENVS]


def _first_chunk_pairwise_rms(records: list[dict[str, object]]) -> list[float]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(int(record["root_seed"]), []).append(record)
    values = []
    for root_seed in sorted(grouped):
        group = sorted(grouped[root_seed], key=lambda item: int(item["replica"]))
        for first, second in combinations(group, 2):
            values.append(
                candidate_rms(
                    np.asarray(first["first_normalized_action_chunk"]),
                    np.asarray(second["first_normalized_action_chunk"]),
                )
            )
    return values


def main(cfg: Config) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(cfg.local_checkpoint_path)
    if checkpoint.name != "trc7rbt0_step_110000":
        raise ValueError("H63 is locked to checkpoint trc7rbt0_step_110000")

    formal = cfg.mode == "audit"
    root_seeds = ROOT_SEEDS if formal else ROOT_SEEDS[:2]
    replicas = FORMAL_REPLICAS if formal else SMOKE_REPLICAS
    device = torch.device(cfg.device)
    actor = load_base_policy(
        str(checkpoint),
        device=device,
        sampling_steps=SAMPLING_STEPS,
        n_action_steps=ACTION_STEPS,
    )
    if actor.config.horizon != 16 or actor.config.n_action_steps != ACTION_STEPS:
        raise ValueError("H63 requires 16-action prediction and execution horizons")
    anchor = snapshot_parameters(actor)

    config_payload = {
        **asdict(cfg),
        "hypothesis": "H63",
        "checkpoint_name": checkpoint.name,
        "root_seeds": list(root_seeds),
        "replicas": replicas,
        "episode_horizon": HORIZON_STEPS,
        "sampling_steps": SAMPLING_STEPS,
        "action_steps": ACTION_STEPS,
        "renderer_claim": "OSMesa mechanism audit; not an official EGL benchmark",
    }
    _atomic_json(output_dir / "config.json", config_payload)

    records = []
    source_sum = 0.0
    source_square_sum = 0.0
    source_count = 0
    for batch_index, scenarios in enumerate(_scenario_batches(root_seeds, replicas)):
        logger.info("Running episode batch %d with %d lanes", batch_index, len(scenarios))
        batch_records, batch_sum, batch_square_sum, batch_count = _rollout_batch(
            actor, cfg, scenarios
        )
        records.extend(batch_records)
        source_sum += batch_sum
        source_square_sum += batch_square_sum
        source_count += batch_count
        _atomic_json(output_dir / "records.json", records)

    if source_count == 0:
        raise RuntimeError("H63 generated no Gaussian source values")
    source_mean = source_sum / source_count
    source_variance = max(0.0, source_square_sum / source_count - source_mean**2)
    results = summarize_group_relative_audit(
        records,
        expected_group_size=replicas,
        expected_group_count=len(root_seeds),
        source_mean=source_mean,
        source_std=math.sqrt(source_variance),
        first_chunk_pairwise_rms=_first_chunk_pairwise_rms(records),
        policy_parameters_bitwise_unchanged=anchor_restored_exactly(actor, anchor),
        formal=formal,
    )
    results["source_value_count"] = source_count
    _atomic_json(output_dir / "results.json", results)
    logger.info("results=%s", json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Config, config=(tyro.conf.FlagConversionOff,)))

