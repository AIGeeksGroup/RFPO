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
from src.binary_gripper_projection import (
    analyze_binary_gripper_projection,
    project_binary_gripper,
)
from src.dexmg_env import create_vectorized_env
from src.parameter_space_es import batched_observation_hashes


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ENVIRONMENT_SEEDS = tuple(range(20261010, 20261030))


@dataclass
class Config:
    local_checkpoint_path: str
    control_results_path: str
    output_dir: str
    mode: Literal["smoke", "audit"] = "smoke"
    device: str = "cuda"
    debug: bool = False
    camera_size: int = 84


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _load_control_hashes(path: Path, environment_seeds: tuple[int, ...]) -> list[str]:
    payload = json.loads(path.read_text())
    control = next(
        record for record in payload["records"] if int(record["phase"]) == 8
    )
    hashes_by_seed = {
        int(episode["environment_seed"]): observation_hash
        for episode, observation_hash in zip(
            control["episodes"], control["initial_observation_hashes"]
        )
    }
    return [hashes_by_seed[seed] for seed in environment_seeds]


def _run_candidate(policy, cfg: Config, environment_seeds: tuple[int, ...]) -> dict[str, object]:
    image_keys = [key.replace("observation.images.", "") for key in policy.config.image_features]
    env = create_vectorized_env(
        env_name="Can",
        num_envs=len(environment_seeds),
        device="cpu" if cfg.device == "cpu" else "cuda",
        camera_size=cfg.camera_size,
        debug=cfg.debug,
        expected_image_keys=image_keys,
        seeds=list(environment_seeds),
    )
    try:
        policy.init_action_buffers(len(environment_seeds))
        policy.reset()
        obs, _ = env.reset()
        with torch.inference_mode():
            normalized_obs = policy.normalize_inputs(
                {key: value.clone() for key, value in obs.items()}
            )
        initial_hashes = batched_observation_hashes(normalized_obs)
        completed = np.zeros(len(environment_seeds), dtype=bool)
        successes = np.zeros(len(environment_seeds), dtype=np.int64)
        returns = np.zeros(len(environment_seeds), dtype=np.float64)
        lengths = np.zeros(len(environment_seeds), dtype=np.int64)
        raw_actions: list[np.ndarray] = []
        projected_actions: list[np.ndarray] = []
        started = time.perf_counter()

        while not completed.all():
            with torch.inference_mode():
                raw_action, _ = policy.select_action(obs, zero_sampling=True)
                projected_action = project_binary_gripper(raw_action)
            active = ~completed
            active_ids = torch.as_tensor(
                np.flatnonzero(active), device=raw_action.device
            )
            raw_actions.append(raw_action[active_ids].detach().cpu().numpy())
            projected_actions.append(
                projected_action[active_ids].detach().cpu().numpy()
            )
            obs, reward, terminated, truncated, _ = env.step(projected_action)
            reward_cpu = reward.detach().cpu().numpy()
            if not np.isfinite(reward_cpu).all():
                raise FloatingPointError("binary gripper audit produced non-finite rewards")
            returns[active] += reward_cpu[active]
            lengths[active] += 1
            done = (terminated | truncated).detach().cpu().numpy()
            newly_done = active & done
            successes[newly_done] = (reward_cpu[newly_done] == 1.0).astype(np.int64)
            completed[newly_done] = True
            done_ids = np.flatnonzero(done)
            if done_ids.size:
                policy.reset(env_ids=torch.as_tensor(done_ids))

        return {
            "initial_observation_hashes": initial_hashes,
            "successes": successes,
            "raw_actions": np.concatenate(raw_actions),
            "projected_actions": np.concatenate(projected_actions),
            "episodes": [
                {
                    "environment_seed": seed,
                    "success": int(successes[index]),
                    "return": float(returns[index]),
                    "length": int(lengths[index]),
                }
                for index, seed in enumerate(environment_seeds)
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        env.close()


def main(cfg: Config) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "audit_results.json"
    checkpoint = Path(cfg.local_checkpoint_path)
    if checkpoint.name != "95j3noe4_step_1000":
        raise ValueError("H56 is locked to checkpoint 95j3noe4_step_1000")

    environment_seeds = ENVIRONMENT_SEEDS if cfg.mode == "audit" else ENVIRONMENT_SEEDS[:2]
    control_hashes = _load_control_hashes(Path(cfg.control_results_path), environment_seeds)
    policy = load_policy(checkpoint, device=cfg.device, load_ema=True)
    if policy.config.horizon != 16 or policy.config.n_action_steps != 8:
        raise ValueError("H56 requires the checkpoint-native 16/8 action horizons")
    policy.config.sampling_steps = 10
    policy.config.integration_method = "euler"
    policy.eval()
    result = _run_candidate(policy, cfg, environment_seeds)
    raw_actions = result.pop("raw_actions")
    projected_actions = result.pop("projected_actions")
    successes = result.pop("successes")
    initial_hashes = result["initial_observation_hashes"]
    analysis = analyze_binary_gripper_projection(
        raw_actions,
        projected_actions,
        successes,
        initial_hashes,
        control_hashes,
        audit_mode=cfg.mode == "audit",
    )
    payload = {
        "metadata": {
            **asdict(cfg),
            "checkpoint_name": checkpoint.name,
            "environment_seeds": list(environment_seeds),
            "sampling_mode": "zero",
            "sampling_steps": 10,
            "integration_method": "euler",
            "prediction_horizon": 16,
            "execution_horizon": 8,
            "projection": "action[..., -1] = +1 if action[..., -1] >= 0 else -1",
            "renderer_claim": "OSMesa method screen; not official EGL or real robot",
        },
        "control_initial_observation_hashes": control_hashes,
        **result,
        "gripper_pair_sample": [
            {"raw": float(raw), "projected": float(projected)}
            for raw, projected in zip(raw_actions[:64, -1], projected_actions[:64, -1])
        ],
        "analysis": analysis,
    }
    _atomic_json(output_path, payload)
    logger.info("analysis=%s", analysis)


if __name__ == "__main__":
    main(tyro.cli(Config, config=(tyro.conf.FlagConversionOff,)))
