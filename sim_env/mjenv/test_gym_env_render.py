import time
import mujoco
import mujoco.viewer
import numpy as np
import mjenv
import gymnasium as gym

# env = gym.make('PandaPickCube-v0', render_mode='rgb_array')
env = gym.make('PandaPickCubeVision-v0', render_mode='human', image_obs=True)
action_spec = env.action_space


def sample():
    a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    return a.astype(action_spec.dtype)


obs, info = env.reset()
frames = []

for i in range(200):
    a = sample()
    obs, rew, done, truncated, info = env.step(a)
    images = obs['images']
    # frames.append(np.concatenate(images, axis=0))
    frames.append(np.concatenate((images['front'], images['wrist']), axis=0))

    if done:
        obs, info = env.reset()

import imageio
imageio.mimsave('test.mp4', frames, fps=20)


