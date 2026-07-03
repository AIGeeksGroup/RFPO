# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import functools, gym, copy
import numpy as np
from glob import glob
from omegaconf import OmegaConf

from hand_imitation.env.create_env import create_env
from hand_imitation.env.gym_wrapper import GymWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder


class InfoCallback(BaseCallback):
    def _on_rollout_end(self) -> None:
        all_keys = {}
        for info in self.model.ep_info_buffer:
            for k, v in info.items():
                if k in ('r', 't', 'l'):
                    continue
                elif k not in all_keys:
                    all_keys[k] = []
                all_keys[k].append(v)

        for k, v in all_keys.items():
            self.model.logger.record(f'env/{k}', np.mean(v))
    
    def _on_step(self) -> bool:
        return True


class _ObsExtractor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.observation_space = env.observation_space
    
    def step(self, action):
        o, r, done, info = self.env.step(action)
        return o, r, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def curriculum(self, stage):
        self.env.env._base_env._stage = stage


class FallbackCheckpoint(BaseCallback):
    def __init__(self, output_dir, checkpoint_freq=1, verbose=0):
        super().__init__(verbose)
        self.output_dir = output_dir
        self.checkpoint_freq = checkpoint_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_freq == 0 or self.n_calls <= 1:
            self.model.save(f'{self.output_dir}/restore_checkpoint')
        return True


def _env_maker(name, norm_traj, task_kwargs, is_eval, info_keywords, robot_name="allegro_hand_ur5"):
    np.random.seed()
    env = create_env(name=name, norm_traj=norm_traj, task_kwargs=task_kwargs, is_eval=is_eval, robot_name=robot_name)
    env = GymWrapper(env)
    env = Monitor(env, info_keywords=tuple(info_keywords))
    env = _ObsExtractor(env)
    return env


def make_env(multi_proc, n_envs, vid_freq, vid_length, **kwargs):
    env_maker = functools.partial(_env_maker, **kwargs)
    if multi_proc:
        env = SubprocVecEnv([env_maker for _ in range(n_envs)])
    else:
        env = DummyVecEnv([env_maker for _ in range(n_envs)])

    if vid_freq is not None:
        vid_freq = max(int(vid_freq // n_envs), 1)
        trigger = lambda x: x % vid_freq == 0 or x <= 1
        env = VecVideoRecorder(env, "videos/", record_video_trigger=trigger, video_length=vid_length)
    return env


def make_policy_kwargs(policy_config):
    return OmegaConf.to_container(policy_config)
