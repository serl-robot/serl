import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation

from serl.wrappers.pixels import wrap_pixels
from serl.wrappers.single_precision import SinglePrecision
from serl.wrappers.universal_seed import UniversalSeed
from serl.wrappers.franka_wrapper import FrankaWrapper
def wrap_gym(env: gym.Env, rescale_actions: bool = True, flatten_states=True) -> gym.Env:
    # env = SinglePrecision(env)
    env = UniversalSeed(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if flatten_states and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env
