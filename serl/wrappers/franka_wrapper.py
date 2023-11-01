import gym
from franka.env_franka.franka_env.envs.wrappers import GripperCloseEnv, SpacemouseIntervention
def FrankaWrapper(env: gym.Env,) -> gym.Env:
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    return env