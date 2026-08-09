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
from src.factorized_source import analyze_factorized_source_audit
from src.parameter_space_es import batched_observation_hashes


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ENVIRONMENT_SEEDS = tuple(range(20261031, 20261051))
SOURCE_BASE_SEED = 20261101


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


def _run_condition(
    policy,
    cfg: Config,
    environment_seeds: tuple[int, ...],
    *,
    condition: Literal["control", "candidate"],
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
        first_actions = None
        started = time.perf_counter()

        while not completed.all():
            with torch.inference_mode():
                action, _ = policy.select_action(
                    obs,
                    stateless_source_seed=SOURCE_BASE_SEED,
                    zero_gripper_source_latent=condition == "candidate",
                )
            if not torch.isfinite(action).all():
                raise FloatingPointError("factorized source audit produced non-finite actions")
            if first_actions is None:
                first_actions = action.detach().cpu().numpy().tolist()
            active = ~completed
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_cpu = reward.detach().cpu().numpy()
            if not np.isfinite(reward_cpu).all():
                raise FloatingPointError("factorized source audit produced non-finite rewards")
            returns[active] += reward_cpu[active]
            lengths[active] += 1
            done = (terminated | truncated).detach().cpu().numpy()
            newly_done = active & done
            successes[newly_done] = (reward_cpu[newly_done] == 1.0).astype(np.int64)
            completed[newly_done] = True

        if first_actions is None:
            raise RuntimeError("factorized source audit generated no actions")
        active_plan_counts = {
            env_id: int((length + policy.config.n_action_steps - 1) // policy.config.n_action_steps)
            for env_id, length in enumerate(lengths)
        }
        source_records = [
            record
            for record in policy.factorized_source_records
            if record["plan_index"] < active_plan_counts[record["environment_id"]]
        ]
        return {
            "condition": condition,
            "initial_observation_hashes": initial_hashes,
            "first_actions": first_actions,
            "episodes": [
                {
                    "environment_seed": seed,
                    "success": int(successes[index]),
                    "return": float(returns[index]),
                    "length": int(lengths[index]),
                }
                for index, seed in enumerate(environment_seeds)
            ],
            "source_records": source_records,
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
        raise ValueError("H57 is locked to checkpoint 95j3noe4_step_1000")

    environment_seeds = ENVIRONMENT_SEEDS if cfg.mode == "audit" else ENVIRONMENT_SEEDS[:2]
    policy = load_policy(checkpoint, device=cfg.device, load_ema=True)
    if policy.config.horizon != 16 or policy.config.n_action_steps != 8:
        raise ValueError("H57 requires the checkpoint-native 16/8 action horizons")
    policy.config.sampling_steps = 10
    policy.config.integration_method = "euler"
    policy.eval()
    payload: dict[str, object] = {
        "metadata": {
            **asdict(cfg),
            "checkpoint_name": checkpoint.name,
            "environment_seeds": list(environment_seeds),
            "source_base_seed": SOURCE_BASE_SEED,
            "sampling_steps": 10,
            "integration_method": "euler",
            "prediction_horizon": 16,
            "execution_horizon": 8,
            "condition_order": ["control", "candidate"],
            "candidate_source": "N(0,1) arm coordinates; zero gripper coordinate",
            "renderer_claim": "OSMesa method screen; not official EGL or real robot",
        },
        "conditions": [],
    }
    for condition in ("control", "candidate"):
        logger.info("condition=%s environments=%d", condition, len(environment_seeds))
        result = _run_condition(
            policy,
            cfg,
            environment_seeds,
            condition=condition,
        )
        payload["conditions"].append(result)  # type: ignore[union-attr]
        _atomic_json(output_path, payload)

    control, candidate = payload["conditions"]
    payload["analysis"] = analyze_factorized_source_audit(
        control,
        candidate,
        audit_mode=cfg.mode == "audit",
    )
    _atomic_json(output_path, payload)
    logger.info("analysis=%s", payload["analysis"])


if __name__ == "__main__":
    main(tyro.cli(Config, config=(tyro.conf.FlagConversionOff,)))
