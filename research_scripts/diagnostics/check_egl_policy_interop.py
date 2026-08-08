"""Check policy inference with a live MuJoCo EGL environment and no DDP."""

import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.dexmg_env import create_vectorized_env
from src.flow_model import FlowMatchingPolicy
from src.flow_model_config import FlowMatchingConfig


def load_policy(checkpoint: Path) -> FlowMatchingPolicy:
    policy_dir = checkpoint / "policy"
    config_payload = json.loads((policy_dir / "config.json").read_text())
    config_payload.pop("type", None)
    config_payload.pop("normalization_mapping", None)
    config = FlowMatchingConfig(**config_payload)
    config.image_features = [key for key in config.input_features if "image" in key]
    config.state_features = [
        key for key in config.input_features if "state" in key or "pos" in key
    ]
    policy = FlowMatchingPolicy(config, dataset_stats=None)
    policy.load_state_dict(load_file(policy_dir / "model.safetensors", device="cpu"))
    policy.config.n_action_steps = 16
    policy.config.sampling_steps = 10
    policy.config.sde_sigma = 0
    return policy.to("cuda").eval()


def main() -> None:
    checkpoint = Path("downloaded_checkpoints/95j3noe4_step_6000")
    policy = load_policy(checkpoint)
    policy.init_action_buffers(1)
    env = create_vectorized_env(
        env_name="Can",
        num_envs=1,
        device="cuda",
        camera_size=84,
        video_key="agentview",
        expected_image_keys=["robot0_eye_in_hand"],
    )
    try:
        observations, _ = env.reset()
        for step in range(16):
            start = time.perf_counter()
            with torch.inference_mode():
                action, path = policy.select_action(observations)
                torch.cuda.synchronize()
            policy_seconds = time.perf_counter() - start

            start = time.perf_counter()
            observations, _, terminated, truncated, _ = env.step(action)
            environment_seconds = time.perf_counter() - start
            done = terminated | truncated
            if done.any():
                policy.reset(env_ids=torch.where(done)[0])
            print(
                "egl-plus-policy",
                step,
                round(policy_seconds, 4),
                round(environment_seconds, 4),
                action.shape,
                path.shape,
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
