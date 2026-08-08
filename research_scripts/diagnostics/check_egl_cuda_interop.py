"""Check whether a MuJoCo EGL context can coexist with CUDA compute."""

import time

import torch

from src.dexmg_env import create_vectorized_env


def main() -> None:
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
        print(
            "env-reset",
            observations["observation.images.robot0_eye_in_hand"].shape,
            flush=True,
        )

        action = torch.zeros((1, 7))
        for step in range(4):
            start = time.perf_counter()
            env.step(action)
            print("env-step", step, round(time.perf_counter() - start, 4), flush=True)

        matrix = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        product = matrix @ matrix
        torch.cuda.synchronize()
        print(
            "egl-plus-cuda",
            round(time.perf_counter() - start, 4),
            float(product.norm()),
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
