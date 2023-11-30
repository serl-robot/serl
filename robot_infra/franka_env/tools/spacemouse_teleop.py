import gymnasium as gym
import franka_env
from tqdm import tqdm
import numpy as np

from franka_env.envs.wrappers import GripperCloseEnv, SpacemouseIntervention

env = gym.make('FrankaRobotiq-Vision-v0')
env = GripperCloseEnv(env)
env = SpacemouseIntervention(env)

env.reset()

# import ipdb; ipdb.set_trace()
for _ in tqdm(range(1000000)):
    obs, rew, done, truncated, info = env.step(action=np.zeros((6,)))
    # print(rew)

    if done:
        env.reset()
