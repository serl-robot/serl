import gym
from robot_infra.env.wrappers import GripperCloseEnv, SpacemouseIntervention
def FrankaWrapper(env: gym.Env,) -> gym.Env:
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    return env