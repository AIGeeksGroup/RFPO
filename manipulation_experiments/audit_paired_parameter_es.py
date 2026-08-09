#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from eval_checkpoint import load_policy
from src.dexmg_env import create_vectorized_env
from src.evaluation_overrides import validate_action_steps
from src.parameter_space_es import (
    PERTURBED_PARAMETER_NAMES,
    anchor_restored_exactly,
    apply_mirrored_direction,
    batched_observation_hashes,
    displacement_manifest,
    evaluate_signal,
    gaussian_direction,
    half_paired_rms,
    parameter_hashes,
    restore_anchor,
    snapshot_parameters,
    validate_parameter_family,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DIRECTION_SEEDS = tuple(range(20261000, 20261016))
ENVIRONMENT_SEEDS = tuple(range(20261020, 20261028))
RELATIVE_SCALE = 0.01


@dataclass
class Config:
    local_checkpoint_path: str
    output_dir: str
    mode: Literal["smoke", "audit"] = "smoke"
    device: str = "cuda"
    debug: bool = False
    camera_size: int = 84


def _first_normalized_chunks(policy, first_paths: torch.Tensor) -> np.ndarray:
    chunks = []
    for env_id in range(policy.num_envs):
        paths = [first_paths[env_id], *list(policy.mdp_x_t_path_buffers[env_id])]
        normalized = torch.stack(paths)[:, -1] * policy.config.actor_scale
        chunks.append(normalized.detach().cpu().numpy())
    return np.stack(chunks)


def _rollout_once(policy, cfg: Config, seeds: tuple[int, ...]) -> dict[str, object]:
    image_keys = [key.replace("observation.images.", "") for key in policy.config.image_features]
    env = create_vectorized_env(
        env_name="Square",
        num_envs=len(seeds),
        device="cpu" if cfg.device == "cpu" else "cuda",
        camera_size=cfg.camera_size,
        debug=cfg.debug,
        expected_image_keys=image_keys,
        seeds=list(seeds),
    )
    try:
        policy.init_action_buffers(len(seeds))
        policy.reset()
        obs, _ = env.reset()
        initial_observation_hashes = batched_observation_hashes(obs)
        returns = np.zeros(len(seeds), dtype=np.float64)
        lengths = np.zeros(len(seeds), dtype=np.int64)
        completed = np.zeros(len(seeds), dtype=bool)
        successes = np.zeros(len(seeds), dtype=np.int64)
        first_chunks = None
        started = time.perf_counter()

        while not completed.all():
            with torch.inference_mode():
                action, paths = policy.select_action(obs, zero_sampling=True)
            if first_chunks is None:
                first_chunks = _first_normalized_chunks(policy, paths)
                if not np.isfinite(first_chunks).all():
                    raise FloatingPointError("policy produced a non-finite normalized action path")
            if not torch.isfinite(action).all():
                raise FloatingPointError("policy produced non-finite actions")
            obs, reward, terminated, truncated, _ = env.step(action)
            active = ~completed
            reward_cpu = reward.detach().cpu().numpy()
            if not np.isfinite(reward_cpu).all():
                raise FloatingPointError("environment produced non-finite rewards")
            returns[active] += reward_cpu[active]
            lengths[active] += 1
            done = (terminated | truncated).detach().cpu().numpy()
            newly_done = active & done
            successes[newly_done] = (reward_cpu[newly_done] == 1.0).astype(np.int64)
            completed[newly_done] = True
            done_ids = np.flatnonzero(done)
            if done_ids.size:
                policy.reset(env_ids=torch.as_tensor(done_ids))

        assert first_chunks is not None
        if not np.isfinite(returns).all():
            raise FloatingPointError("rollout produced non-finite episode returns")
        episodes = [
            {
                "environment_seed": int(seed),
                "success": int(successes[index]),
                "return": float(returns[index]),
                "length": int(lengths[index]),
            }
            for index, seed in enumerate(seeds)
        ]
        return {
            "episodes": episodes,
            "initial_observation_hashes": initial_observation_hashes,
            "first_normalized_action_chunks": first_chunks.tolist(),
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        env.close()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    temporary.replace(path)


def _completed_key(record: dict[str, object]) -> tuple[int, int]:
    return int(record["direction_index"]), int(record["sign"])


def main(cfg: Config) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cfg.mode}_results.json"
    direction_seeds = DIRECTION_SEEDS if cfg.mode == "audit" else DIRECTION_SEEDS[:1]
    environment_seeds = ENVIRONMENT_SEEDS if cfg.mode == "audit" else ENVIRONMENT_SEEDS[:1]

    checkpoint = Path(cfg.local_checkpoint_path)
    if checkpoint.name != "trc7rbt0_step_110000":
        raise ValueError("H53 is locked to checkpoint trc7rbt0_step_110000")
    policy = load_policy(checkpoint, device=cfg.device, load_ema=True)
    policy.config.sampling_steps = 10
    policy.config.integration_method = "euler"
    policy.config.n_action_steps = validate_action_steps(16, policy.config.horizon)
    policy.eval()
    validate_parameter_family(policy)
    anchor = snapshot_parameters(policy)

    payload: dict[str, object]
    if output_path.exists():
        payload = json.loads(output_path.read_text())
        if payload["metadata"]["mode"] != cfg.mode:
            raise ValueError("existing result mode does not match requested mode")
    else:
        payload = {
            "metadata": {
                **asdict(cfg),
                "checkpoint_name": checkpoint.name,
                "direction_seeds": list(direction_seeds),
                "environment_seeds": list(environment_seeds),
                "relative_scale": RELATIVE_SCALE,
                "perturbed_parameters": list(PERTURBED_PARAMETER_NAMES),
                "sampling_steps": 10,
                "integration_method": "euler",
                "action_steps": 16,
                "sampling_mode": "zero",
                "renderer_claim": "OSMesa paired method audit; not official EGL or real robot",
                "common_random_number_method": "recreate vector env for each sign with identical constructor seeds",
            },
            "anchor_parameter_hashes": parameter_hashes(anchor),
            "records": [],
        }
        _atomic_json(output_path, payload)

    records: list[dict[str, object]] = payload["records"]  # type: ignore[assignment]
    completed = {_completed_key(record) for record in records}
    for direction_index, direction_seed in enumerate(direction_seeds):
        direction = gaussian_direction(anchor, direction_seed, RELATIVE_SCALE)
        for sign in (-1, 1):
            if (direction_index, sign) in completed:
                continue
            apply_mirrored_direction(policy, anchor, direction, sign)
            manifest = displacement_manifest(policy, anchor)
            if not manifest["finite"] or not manifest["unlisted_parameters_bitwise_unchanged"]:
                raise RuntimeError("parameter validity check failed before rollout")
            if any(abs(value - RELATIVE_SCALE) > 1e-5 for value in manifest["relative_norms"].values()):
                raise RuntimeError("locked relative perturbation norm check failed")
            logger.info(
                "direction=%d seed=%d sign=%+d envs=%d",
                direction_index, direction_seed, sign, len(environment_seeds),
            )
            rollout = _rollout_once(policy, cfg, environment_seeds)
            post_rollout_manifest = displacement_manifest(policy, anchor)
            if post_rollout_manifest != manifest:
                raise RuntimeError("policy parameters changed during inference")
            current_parameters = dict(policy.named_parameters())
            record = {
                "direction_index": direction_index,
                "direction_seed": direction_seed,
                "sign": sign,
                "environment_seeds": list(environment_seeds),
                "perturbation": manifest,
                "post_rollout_perturbation": post_rollout_manifest,
                "perturbed_parameter_hashes": parameter_hashes(
                    current_parameters, PERTURBED_PARAMETER_NAMES
                ),
                **rollout,
            }
            records.append(record)
            _atomic_json(output_path, payload)
            restore_anchor(policy, anchor)
            if not anchor_restored_exactly(policy, anchor):
                raise RuntimeError("anchor restore was not bitwise exact")

    restore_anchor(policy, anchor)
    payload["anchor_restore_bitwise_exact"] = anchor_restored_exactly(policy, anchor)
    if cfg.mode == "audit":
        by_key = {_completed_key(record): record for record in records}
        successes = np.zeros((16, 2, 8), dtype=np.float64)
        action_rms = []
        for direction_index in range(16):
            negative = by_key[(direction_index, -1)]
            positive = by_key[(direction_index, 1)]
            for sign_index, record in enumerate((negative, positive)):
                successes[direction_index, sign_index] = [
                    episode["success"] for episode in record["episodes"]
                ]
            action_rms.append(
                half_paired_rms(
                    np.asarray(positive["first_normalized_action_chunks"]),
                    np.asarray(negative["first_normalized_action_chunks"]),
                )
            )
        signal = evaluate_signal(successes)
        median_action_rms = float(np.median(action_rms))
        common_random_initial_observations_exact = all(
            by_key[(direction_index, -1)]["initial_observation_hashes"]
            == by_key[(direction_index, 1)]["initial_observation_hashes"]
            for direction_index in range(16)
        )
        payload["analysis"] = {
            **signal,
            "first_action_half_paired_rms_by_direction": action_rms,
            "median_first_action_half_paired_rms": median_action_rms,
            "action_rms_valid": 0.01 <= median_action_rms <= 0.10,
            "common_random_initial_observations_exact": common_random_initial_observations_exact,
            "complete_episode_count": int(successes.size),
            "all_validity_gates_pass": bool(
                payload["anchor_restore_bitwise_exact"]
                and 0.01 <= median_action_rms <= 0.10
                and common_random_initial_observations_exact
                and all(
                    record["perturbation"]["finite"]
                    and record["perturbation"]["unlisted_parameters_bitwise_unchanged"]
                    and len(record["episodes"]) == 8
                    for record in records
                )
            ),
        }
    _atomic_json(output_path, payload)
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    main(tyro.cli(Config))
