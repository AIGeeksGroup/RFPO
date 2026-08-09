#!/usr/bin/env python

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from src.deterministic_reference import (
    summarize_deterministic_reference_audit,
    validate_gaussian_manifest,
)
from src.dexmg_env import create_vectorized_env
from src.parameter_space_es import (
    anchor_restored_exactly,
    batched_observation_hashes,
    snapshot_parameters,
)
from src.vine_returns import array_sha256
from train_residual_flow_steering import (
    action_bounds,
    load_base_policy,
    subset_observations,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

ROOT_SEEDS = tuple(range(20261080, 20261088))
ARCHIVE_SHA256 = "d6e6568f2117f88ebb15239a589853882b1653a48471591abac20689b249003c"
HORIZON_STEPS = 400
SAMPLING_STEPS = 10
ACTION_STEPS = 16


@dataclass
class Config:
    local_checkpoint_path: str
    archived_records_path: str
    output_dir: str
    mode: Literal["smoke", "audit"] = "smoke"
    device: str = "cuda"
    camera_size: int = 84


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_env(actor, seeds: tuple[int, ...], cfg: Config):
    image_keys = [
        key.replace("observation.images.", "") for key in actor.config.image_features
    ]
    return create_vectorized_env(
        env_name="Square",
        num_envs=len(seeds),
        device=cfg.device,
        camera_size=cfg.camera_size,
        video_key="agentview",
        expected_image_keys=image_keys,
        seeds=list(seeds),
    )


@torch.no_grad()
def _rollout_references(actor, cfg: Config, root_seeds: tuple[int, ...]):
    device = torch.device(cfg.device)
    env = _make_env(actor, root_seeds, cfg)
    low, high = action_bounds(env, device)
    observations, _ = env.reset()
    initial_hashes = batched_observation_hashes(observations)
    privileged_hashes = [
        array_sha256(np.asarray(value)) for value in env.call("get_privileged_state")
    ]
    completed = np.zeros(len(root_seeds), dtype=bool)
    successes = np.zeros(len(root_seeds), dtype=np.int64)
    lengths = np.zeros(len(root_seeds), dtype=np.int64)
    chunks: list[torch.Tensor | None] = [None] * len(root_seeds)
    offsets = [0] * len(root_seeds)
    plan_indices = [0] * len(root_seeds)
    first_normalized_chunks: list[list[list[float]] | None] = [None] * len(root_seeds)
    source_value_count = 0
    source_nonzero_count = 0

    try:
        for _ in range(HORIZON_STEPS):
            replan_lanes = [
                lane
                for lane in range(len(root_seeds))
                if not completed[lane] and chunks[lane] is None
            ]
            if replan_lanes:
                indices = torch.as_tensor(replan_lanes, device=device, dtype=torch.long)
                cpu_sources = torch.zeros(
                    (
                        len(replan_lanes),
                        actor.config.horizon,
                        actor.model.action_dim,
                    ),
                    dtype=torch.float32,
                    device="cpu",
                )
                source_value_count += cpu_sources.numel()
                source_nonzero_count += int(torch.count_nonzero(cpu_sources).item())
                predicted, _ = actor.predict_action_chunk(
                    subset_observations(observations, indices),
                    source_noise=cpu_sources.to(device),
                )
                predicted = predicted[:, :ACTION_STEPS].clamp(low, high)
                normalized = actor.normalize_targets({"action": predicted.clone()})[
                    "action"
                ]
                for local_index, lane in enumerate(replan_lanes):
                    chunks[lane] = predicted[local_index]
                    offsets[lane] = 0
                    plan_indices[lane] += 1
                    if first_normalized_chunks[lane] is None:
                        first_normalized_chunks[lane] = (
                            normalized[local_index].cpu().tolist()
                        )

            actions = []
            for lane in range(len(root_seeds)):
                if completed[lane]:
                    actions.append(torch.zeros_like(low))
                    continue
                chunk = chunks[lane]
                if chunk is None:
                    raise RuntimeError("active reference lane has no action chunk")
                actions.append(chunk[offsets[lane]])

            observations, rewards, terminated, truncated, _ = env.step(
                torch.stack(actions)
            )
            reward_values = rewards.detach().cpu().numpy()
            if not np.isfinite(reward_values).all():
                raise FloatingPointError(
                    "deterministic-reference rollout produced non-finite rewards"
                )
            active = ~completed
            lengths[active] += 1
            done_values = (terminated | truncated).detach().cpu().numpy()
            newly_done = active & done_values
            successes[newly_done] = (reward_values[newly_done] == 1.0).astype(np.int64)
            completed[newly_done] = True

            for lane in range(len(root_seeds)):
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
    for lane, root_seed in enumerate(root_seeds):
        first_chunk = first_normalized_chunks[lane]
        if first_chunk is None:
            raise RuntimeError("reference episode produced no action chunk")
        records.append(
            {
                "root_seed": root_seed,
                "source": "zero",
                "success": int(successes[lane]),
                "completed": bool(completed[lane]),
                "length": int(lengths[lane]),
                "plan_count": int(plan_indices[lane]),
                "initial_observation_hash": initial_hashes[lane],
                "initial_privileged_state_hash": privileged_hashes[lane],
                "first_normalized_action_chunk": first_chunk,
            }
        )
    return records, source_value_count, source_nonzero_count


def main(cfg: Config) -> None:
    output_dir = Path(cfg.output_dir)
    checkpoint = Path(cfg.local_checkpoint_path)
    archive_path = Path(cfg.archived_records_path)
    if checkpoint.name != "trc7rbt0_step_110000":
        raise ValueError("H64 is locked to checkpoint trc7rbt0_step_110000")

    archive_sha256 = _file_sha256(archive_path)
    if archive_sha256 != ARCHIVE_SHA256:
        raise ValueError(
            f"archived H63 checksum mismatch: {archive_sha256} != {ARCHIVE_SHA256}"
        )
    with archive_path.open() as handle:
        archived_records = json.load(handle)
    if not isinstance(archived_records, list):
        raise ValueError("archived H63 records must be a JSON list")
    full_manifest = validate_gaussian_manifest(
        archived_records,
        expected_root_seeds=ROOT_SEEDS,
        replicas=4,
    )
    archive_manifest_exact = True

    formal = cfg.mode == "audit"
    root_seeds = ROOT_SEEDS if formal else ROOT_SEEDS[:2]
    selected_gaussian = [
        record for seed in root_seeds for record in full_manifest[seed]
    ]
    device = torch.device(cfg.device)
    actor = load_base_policy(
        str(checkpoint),
        device=device,
        sampling_steps=SAMPLING_STEPS,
        n_action_steps=ACTION_STEPS,
    )
    if actor.config.horizon != 16 or actor.config.n_action_steps != ACTION_STEPS:
        raise ValueError("H64 requires 16-action prediction and execution horizons")
    anchor = snapshot_parameters(actor)

    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        output_dir / "config.json",
        {
            **asdict(cfg),
            "hypothesis": "H64",
            "checkpoint_name": checkpoint.name,
            "root_seeds": list(root_seeds),
            "episode_horizon": HORIZON_STEPS,
            "sampling_steps": SAMPLING_STEPS,
            "action_steps": ACTION_STEPS,
            "archived_records_sha256": archive_sha256,
            "locked_archived_records_sha256": ARCHIVE_SHA256,
            "renderer_claim": "OSMesa mechanism audit; not an official EGL benchmark",
        },
    )
    reference_records, source_value_count, source_nonzero_count = _rollout_references(
        actor, cfg, root_seeds
    )
    _atomic_json(output_dir / "records.json", reference_records)
    results = summarize_deterministic_reference_audit(
        selected_gaussian,
        reference_records,
        expected_root_seeds=root_seeds,
        archive_manifest_exact=archive_manifest_exact,
        zero_sources_bitwise_exact=(
            source_value_count > 0 and source_nonzero_count == 0
        ),
        policy_parameters_bitwise_unchanged=anchor_restored_exactly(actor, anchor),
        formal=formal,
    )
    results["archived_records_sha256"] = archive_sha256
    results["source_value_count"] = source_value_count
    results["source_nonzero_count"] = source_nonzero_count
    _atomic_json(output_dir / "results.json", results)
    logger.info("results=%s", json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Config, config=(tyro.conf.FlagConversionOff,)))
