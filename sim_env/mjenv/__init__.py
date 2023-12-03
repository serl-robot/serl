from mjenv.core import MujocoDmEnv, RenderingSpec
from mjenv.mujoco_gym_env import MujocoGymEnv, GymRenderingSpec
__all__ = [
    "MujocoDmEnv",
    "RenderingSpec",
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gymnasium.envs.registration import register
register(
    id='PandaPickCube-v0',
    entry_point='mjenv.envs:PandaPickCubeGymEnv',
    max_episode_steps=100,
)
register(
    id='PandaPickCubeVision-v0',
    entry_point='mjenv.envs:PandaPickCubeGymEnv',
    max_episode_steps=100,
    kwargs={'image_obs': True},
)