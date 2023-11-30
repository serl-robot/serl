from typing import Optional, Sequence
from collections import OrderedDict
import gymnasium as gym
import numpy as np
import sys
import wandb

class WANDBVideo(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        name: str = "video",
        render_kwargs={},
        max_videos: Optional[int] = None,
        pixel_keys: Sequence[str] = ("pixels",),
    ):
        super().__init__(env)
        self._name = name
        self._render_kwargs = render_kwargs
        self._max_videos = max_videos
        self._video = OrderedDict()
        self._rendered_video = []
        self._rewards = []
        self._pixel_keys = pixel_keys

    def get_rendered_video(self):
        rendered_video = [np.array(v) for v in self._video.values()]
        rendered_video = np.concatenate(rendered_video, axis=1)
        if rendered_video.ndim == 5:
            rendered_video = rendered_video[..., -1]
        return rendered_video

    def get_video(self):
        video = {k: np.array(v) for k,v in self._video.items()}
        return video

    def get_rewards(self):
        return self._rewards

    def _add_frame(self, obs):
        if self._max_videos is not None and self._max_videos <= 0:
            return
        if isinstance(obs, dict):
            img = []
            for k in self._pixel_keys:
                if k in obs:
                    if k in self._video:
                        self._video[k].append(obs[k])
                    else:
                        self._video[k] = [obs[k]]
        else:
            raise Exception("bad obs")
            self._video.append(
                self.render(
                    height=self._pixel_hw,
                    width=self._pixel_hw,
                    mode="rgb_array",
                    **self._render_kwargs
                )
            )

    def _add_rewards(self, rew):
        self._rewards.append(rew)

    def reset(self, **kwargs):
        self._video.clear()
        self._rendered_video.clear()
        self._rewards.clear()
        obs, info = super().reset(**kwargs)
        self._add_frame(obs)
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, done, truncate, info = super().step(action)
        self._add_frame(obs)
        self._add_rewards(reward)

        if done and len(self._video) > 0:
            if self._max_videos is not None:
                self._max_videos -= 1
            video = self.get_rendered_video().transpose(0, 3, 1, 2)
            if video.shape[1] == 1:
                video = np.repeat(video, 3, 1)
            video = wandb.Video(video, fps=10, format="mp4")
            wandb.log({self._name: video}, commit=False)

        return obs, reward, done, truncate, info
