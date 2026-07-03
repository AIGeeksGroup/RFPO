
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from gym import core, spaces
import numpy as np
import collections


class GymWrapper(core.Env):
    metadata = {"render.modes": ['rgb_array'], "video.frames_per_second": 25}

    def __init__(self, base_env):
        """
        Initializes 
        """
        self._base_env = base_env

        if base_env.is_vision and not base_env.is_demo_rollout:
            # parses and stores action space
            self.action_space = spaces.Box(low=-1, high=1, shape=(base_env.action_dim,), dtype=np.float32)
            state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(base_env.obs_dim,))
            point_cloud_shape = (512, 3)
            point_low = -np.inf * np.ones(point_cloud_shape)
            point_high = np.inf * np.ones(point_cloud_shape)
            point_cloud_space = spaces.Box(low=point_low, high=point_high, dtype=np.float32)
            self.observation_space = spaces.Dict({"agent_pos": state_space, "point_cloud": point_cloud_space})
        else:
            # parses and stores action space
            self.action_space = spaces.Box(low=-1, high=1, shape=(base_env.action_dim,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(base_env.obs_dim,))

    def reset(self):
        obs = self._base_env.reset()
        return obs
    
    def step(self, action):
        o, r, done, info = self._base_env.step(action.astype(self.action_space.dtype))
        return o, r, done, info

    def is_success(self):
        return self._base_env.is_success()

    def render(self, camera_id=None, mode=None):
        # self._base_env.scene.step()
        self._base_env.scene.update_render()
        try:
            self._base_env.cameras['relocate_viz'].take_picture()
            img = self._base_env.cameras['relocate_viz'].get_float_texture('Color')[:, :, :3]
        except:
            self._base_env.cameras['instance_1'].take_picture()
            img = self._base_env.cameras['instance_1'].get_float_texture('Color')[:, :, :3]
        img = (img * 255).clip(0, 255).astype("uint8")
        return img

    @property
    def wrapped(self):
        return self._base_env
