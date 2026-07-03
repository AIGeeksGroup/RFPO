import copy
import functools

import gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

from hand_imitation.env.create_rm75_env import create_rm75_env
from hand_imitation.env.gym_wrapper import GymWrapper


class RM75ObsExtractor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.observation_space = env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def curriculum(self, stage):
        self.env.env._base_env._stage = stage


def _rm75_env_maker(name, norm_traj, task_kwargs, is_eval, info_keywords, robot_name="rm75_inspire_right"):
    np.random.seed()
    env = create_rm75_env(
        name=name,
        norm_traj=norm_traj,
        task_kwargs=task_kwargs,
        is_eval=is_eval,
        robot_name=robot_name,
    )
    env = GymWrapper(env)
    env = Monitor(env, info_keywords=tuple(info_keywords))
    env = RM75ObsExtractor(env)
    return env


def make_rm75_env(multi_proc, n_envs, vid_freq, vid_length, **kwargs):
    env_maker = functools.partial(_rm75_env_maker, **kwargs)
    if multi_proc:
        env = SubprocVecEnv([env_maker for _ in range(n_envs)])
    else:
        env = DummyVecEnv([env_maker for _ in range(n_envs)])

    if vid_freq is not None:
        vid_freq = max(int(vid_freq // n_envs), 1)
        trigger = lambda x: x % vid_freq == 0 or x <= 1
        env = VecVideoRecorder(env, "videos/", record_video_trigger=trigger, video_length=vid_length)
    return env


def make_rm75_eval_env(multi_proc, n_eval_envs, **kwargs):
    eval_args = copy.deepcopy(kwargs)
    eval_args["multi_proc"] = multi_proc
    eval_args["is_eval"] = True
    eval_args["n_envs"] = n_eval_envs
    eval_args["vid_freq"] = None
    if "rand_reset_prob" in eval_args.get("task_kwargs", {}):
        eval_args["task_kwargs"]["rand_reset_prob"] = 0
    env = make_rm75_env(**eval_args)
    env.has_multiproc = multi_proc
    return env
