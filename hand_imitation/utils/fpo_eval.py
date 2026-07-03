from __future__ import annotations

import os
import time
from typing import Any

import imageio
import numpy as np

try:
    from stable_baselines3.common.evaluation import evaluate_policy
except ModuleNotFoundError:
    evaluate_policy = None


class FPOEvalCallback:
    def __init__(
        self,
        output_dir: str,
        n_eval_episodes: int = 25,
        fps: int = 25,
        deterministic: bool = False,
    ):
        self.output_dir = output_dir
        self.n_eval_episodes = int(n_eval_episodes)
        self.fps = int(fps)
        self.deterministic = bool(deterministic)
        self.record_video = os.environ.get("VIVIDEX_HEADLESS_NO_RENDER", "0") != "1"
        self.video_dir = os.path.join(output_dir, "eval_videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self._info_tracker: dict[str, list[Any]] = {}
        self._step_info_tracker: dict[str, list[Any]] = {}

    def _render_eval_env(self, env):
        if getattr(env, "has_multiproc", False):
            pipe = env.remotes[0]
            pipe.send(("render", "rgb_array"))
            return pipe.recv()
        return env.envs[0].render("rgb_array")

    def _info_callback(self, locals_, _globals):
        if self.record_video and locals_["i"] == 0:
            try:
                frame = self._render_eval_env(locals_["env"])
                self._info_tracker.setdefault("rollout_video", []).append(frame)
            except Exception:
                pass

        info = locals_["info"]
        for key, value in info.items():
            if isinstance(value, (float, int, np.integer, np.floating, bool)):
                self._step_info_tracker.setdefault(key, []).append(float(value))

        if locals_["done"]:
            for key, value in info.items():
                if isinstance(value, (float, int, np.integer, np.floating, bool)):
                    self._info_tracker.setdefault(key, []).append(float(value))

    def __call__(self, *, model, eval_env, num_timesteps: int) -> dict[str, float]:
        if evaluate_policy is None:
            raise ModuleNotFoundError("stable_baselines3 is required to evaluate FPO policies")
        self._info_tracker = {}
        self._step_info_tracker = {}
        start = time.time()
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=False,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=True,
            callback=self._info_callback,
        )
        metrics: dict[str, float] = {
            "eval/time": float(time.time() - start),
            "eval/mean_reward": float(np.mean(episode_rewards)),
            "eval/mean_length": float(np.mean(episode_lengths)),
        }
        for key, values in self._info_tracker.items():
            if key == "rollout_video":
                if len(values) > 0:
                    path = os.path.join(self.video_dir, f"eval-step-{num_timesteps}.mp4")
                    writer = imageio.get_writer(path, fps=self.fps)
                    for frame in values:
                        writer.append_data(frame)
                    writer.close()
                continue
            if len(values) > 0:
                metrics[f"eval/mean_{key}"] = float(np.mean(values))
        for key, values in self._step_info_tracker.items():
            if len(values) > 0:
                metrics[f"eval/rollout_mean_{key}"] = float(np.mean(values))
        return metrics
