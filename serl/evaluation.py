from typing import Dict

import gym
import numpy as np

from serl.wrappers.wandb_video import WANDBVideo

def evaluate(agent,
             env: gym.Env,
             num_episodes: int,
             save_video: bool = False,
             name='eval_video',
             reset_kwargs={}) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name=name, max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for i in range(num_episodes):
        observation, info = env.reset(**reset_kwargs)
        done = False
        while not done:
            action = agent.eval_actions(observation)
            observation, rew, done, truncated, info = env.step(action)
            done = done or truncated

    return {
        'return': np.mean(env.return_queue),
        'length': np.mean(env.length_queue)
    }, env.get_video() if save_video else []
