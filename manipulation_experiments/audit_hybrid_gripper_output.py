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
from src.hybrid_gripper_output import analyze_hybrid_gripper_audit
from src.parameter_space_es import batched_observation_hashes


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ENVIRONMENT_SEEDS = tuple(range(20261031, 20261051))
SOURCE_BASE_SEED = 20261101


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


def _load_control(
    path: Path, environment_seeds: tuple[int, ...]
) -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(path.read_text())
    metadata = payload["metadata"]
    if metadata["environment_seeds"] != list(environment_seeds):
        raise ValueError("H58 control environment seeds do not match the locked condition")
    if metadata["source_base_seed"] != SOURCE_BASE_SEED:
        raise ValueError("H58 control source seed does not match the locked condition")
    control = next(
        condition
        for condition in payload["conditions"]
        if condition["condition"] == "control"
    )
    return metadata, control


def _run_candidate(
    policy, cfg: Config, environment_seeds: tuple[int, ...]
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
                    deterministic_gripper_output=True,
                )
            if not torch.isfinite(action).all():
                raise FloatingPointError("hybrid gripper audit produced non-finite actions")
            if first_actions is None:
                first_actions = action.detach().cpu().numpy().tolist()
            active = ~completed
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_cpu = reward.detach().cpu().numpy()
            if not np.isfinite(reward_cpu).all():
                raise FloatingPointError("hybrid gripper audit produced non-finite rewards")
            returns[active] += reward_cpu[active]
            lengths[active] += 1
            done = (terminated | truncated).detach().cpu().numpy()
            newly_done = active & done
            successes[newly_done] = (reward_cpu[newly_done] == 1.0).astype(np.int64)
            completed[newly_done] = True

        if first_actions is None:
            raise RuntimeError("hybrid gripper audit generated no actions")
        active_plan_counts = {
            env_id: int((length + policy.config.n_action_steps - 1) // policy.config.n_action_steps)
            for env_id, length in enumerate(lengths)
        }

        def active_record(record: dict[str, object]) -> bool:
            return record["plan_index"] < active_plan_counts[record["environment_id"]]

        return {
            "condition": "candidate",
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
            "source_records": [
                record
                for record in policy.factorized_source_records
                if active_record(record)
            ],
            "hybrid_output_records": [
                record
                for record in policy.hybrid_gripper_output_records
                if active_record(record)
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
        raise ValueError("H58 is locked to checkpoint 95j3noe4_step_1000")

    environment_seeds = ENVIRONMENT_SEEDS if cfg.mode == "audit" else ENVIRONMENT_SEEDS[:2]
    control_metadata, control = _load_control(
        Path(cfg.control_results_path), environment_seeds
    )
    policy = load_policy(checkpoint, device=cfg.device, load_ema=True)
    if policy.config.horizon != 16 or policy.config.n_action_steps != 8:
        raise ValueError("H58 requires the checkpoint-native 16/8 action horizons")
    policy.config.sampling_steps = 10
    policy.config.integration_method = "euler"
    policy.eval()
    candidate = _run_candidate(policy, cfg, environment_seeds)
    analysis = analyze_hybrid_gripper_audit(
        control,
        candidate,
        audit_mode=cfg.mode == "audit",
    )
    payload = {
        "metadata": {
            **asdict(cfg),
            "checkpoint_name": checkpoint.name,
            "environment_seeds": list(environment_seeds),
            "source_base_seed": SOURCE_BASE_SEED,
            "sampling_steps_per_solve": 10,
            "flow_solves_per_replan": 2,
            "integration_method": "euler",
            "prediction_horizon": 16,
            "execution_horizon": 8,
            "control_metadata": control_metadata,
            "candidate_output": "Gaussian arm coordinates; zero-source gripper coordinate",
            "renderer_claim": "OSMesa method screen; not official EGL or real robot",
        },
        "control_reference": {
            "results_path": cfg.control_results_path,
            "successes": sum(episode["success"] for episode in control["episodes"]),
            "episodes": len(control["episodes"]),
        },
        "candidate": candidate,
        "analysis": analysis,
    }
    _atomic_json(output_path, payload)
    logger.info("analysis=%s", analysis)


if __name__ == "__main__":
    main(tyro.cli(Config, config=(tyro.conf.FlagConversionOff,)))
