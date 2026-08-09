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
from src.parameter_space_es import batched_observation_hashes
from src.replanning_phase import analyze_phase_audit


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PHASES = (8, 1, 2, 3, 4, 5, 6, 7)
ENVIRONMENT_SEEDS = tuple(range(20261010, 20261030))


@dataclass
class Config:
    local_checkpoint_path: str
    output_dir: str
    mode: Literal["smoke", "audit"] = "smoke"
    device: str = "cuda"
    debug: bool = False
    camera_size: int = 84


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _rollout_phase(
    policy,
    cfg: Config,
    phase: int,
    environment_seeds: tuple[int, ...],
) -> dict[str, object]:
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
        started = time.perf_counter()

        while not completed.all():
            with torch.inference_mode():
                action, _ = policy.select_action(
                    obs,
                    zero_sampling=True,
                    initial_action_steps=phase,
                )
            if not torch.isfinite(action).all():
                raise FloatingPointError("phase audit produced non-finite actions")
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_cpu = reward.detach().cpu().numpy()
            if not np.isfinite(reward_cpu).all():
                raise FloatingPointError("phase audit produced non-finite rewards")
            active = ~completed
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
            "phase": phase,
            "initial_observation_hashes": initial_hashes,
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
        raise ValueError("H55 is locked to checkpoint 95j3noe4_step_1000")

    policy = load_policy(checkpoint, device=cfg.device, load_ema=True)
    if policy.config.horizon != 16 or policy.config.n_action_steps != 8:
        raise ValueError("H55 requires the checkpoint-native 16/8 action horizons")
    policy.config.sampling_steps = 10
    policy.config.integration_method = "euler"
    policy.eval()
    phases = PHASES if cfg.mode == "audit" else PHASES[:2]
    environment_seeds = (
        ENVIRONMENT_SEEDS if cfg.mode == "audit" else ENVIRONMENT_SEEDS[:2]
    )
    payload: dict[str, object] = {
        "metadata": {
            **asdict(cfg),
            "checkpoint_name": checkpoint.name,
            "phases": list(phases),
            "environment_seeds": list(environment_seeds),
            "sampling_mode": "zero",
            "sampling_steps": 10,
            "integration_method": "euler",
            "renderer_claim": "OSMesa mechanism screen; not official EGL or real robot",
        },
        "records": [],
    }
    for phase in phases:
        logger.info("phase=%d environments=%d", phase, len(environment_seeds))
        record = _rollout_phase(policy, cfg, phase, environment_seeds)
        payload["records"].append(record)  # type: ignore[union-attr]
        _atomic_json(output_path, payload)

    records = payload["records"]
    outcomes = np.asarray(
        [[episode["success"] for episode in record["episodes"]] for record in records]
    )
    hashes = [record["initial_observation_hashes"] for record in records]
    payload["outcome_matrix"] = outcomes.tolist()
    payload["analysis"] = analyze_phase_audit(outcomes, hashes, phases)
    _atomic_json(output_path, payload)
    logger.info("analysis=%s", payload["analysis"])


if __name__ == "__main__":
    main(tyro.cli(Config, config=(tyro.conf.FlagConversionOff,)))
