import collections

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack: int, stacking_key: str = "pixels"):
        super().__init__(env)
        self._num_stack = num_stack
        self._stacking_key = stacking_key

        for key in stacking_key:
            assert key in self.observation_space.spaces
            pixel_obs_spaces = self.observation_space.spaces[key]
            self._env_dim = pixel_obs_spaces.shape[-1]
            low = np.repeat(pixel_obs_spaces.low[..., np.newaxis], num_stack, axis=-1)
            high = np.repeat(pixel_obs_spaces.high[..., np.newaxis], num_stack, axis=-1)
            new_pixel_obs_spaces = Box(low=low, high=high, dtype=pixel_obs_spaces.dtype)
            self.observation_space.spaces[key] = new_pixel_obs_spaces

        self._frames = collections.deque(maxlen=num_stack)

    def reset(self):
        obs, info = self.env.reset()
        for i in range(self._num_stack):
            self._frames.append({key: obs[key] for key in self._stacking_key})
        for k in self._stacking_key:
            obs[k] = self.frames[k]
        return obs, info

    @property
    def frames(self):
        tmp = {}
        for k in self._stacking_key:
            tmp[k] = np.stack([frame[k] for frame in self._frames], axis=-1)
        return tmp

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._frames.append({k: obs[k] for k in self._stacking_key})
        for k in self._stacking_key:
            obs[k] = self.frames[k]
        return obs, reward, done, truncated, info
