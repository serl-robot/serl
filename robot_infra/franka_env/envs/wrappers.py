import time
import numpy as np
import copy

import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.spaces import Box
from gymnasium.spaces import flatten_space, flatten

from franka_env.spacemouse.spacemouse_teleop import SpaceMouseExpert


class FrankaSERLObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            'state': flatten_space(self.env.observation_space['state']),
            **(self.env.observation_space['images'])
        })

    def observation(self, obs):
        obs = {
            'state': flatten(self.env.observation_space['state'], obs['state']),
            **(obs['images'])
        }
        return obs

class GripperCloseEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.ones((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert(
            xyz_dims=3,
            xyz_remap=[0, 1, 2],
            xyz_scale=200,
            rot_scale=200,
            all_angles=True
        )
        self.last_intervene = 0

    def action(self, action: np.ndarray) -> np.ndarray:
        '''
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        '''
        controller_a, _, left, right = self.expert.get_action()
        expert_a = np.zeros((6,))
        if self.gripper_enabled:
            expert_a = np.zeros((7,))
            expert_a[-1] = np.random.uniform(-1, 0)

        expert_a[:3] = controller_a[:3] # XYZ
        expert_a[3] = controller_a[4]  # Roll
        expert_a[4] = controller_a[5] # Pitch
        expert_a[5] = -controller_a[6] # Yaw

        if self.gripper_enabled: #TODO: fix this
            if left:
                expert_a[6] = np.random.uniform(0, 1)
                self.last_intervene = time.time()

            if np.linalg.norm(expert_a[:6]) > 0.001:
                self.last_intervene = time.time()
        else:
            if np.linalg.norm(expert_a) > 0.001:
                self.last_intervene = time.time()

        if time.time() - self.last_intervene < 0.5:
            return expert_a

        return action

class FourDoFWrapper(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def action(self, action):
        a = np.zeros(4)
        a[:3] = action[:3]
        a[-1] = action[-1]
        return a

class TwoCameraFrankaWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        ProxyEnv.__init__(self, env)
        self.env = env
        self.observation_space = spaces.Dict(
            {
                "state": spaces.flatten_space(self.env.observation_space['state_observation']),
                "wrist_1": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                "wrist_2": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                # "side_1": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
            }
        )

    def observation(self, obs):
        ob = {
            'state': spaces.flatten(self.env.observation_space['state_observation'],
                            obs['state_observation']),
            'wrist_1': obs['image_observation']['wrist_1'][...,::-1], # flip color channel
            'wrist_2': obs['image_observation']['wrist_2'][...,::-1], # flip color channel
            # 'side_1': obs['image_observation']['side_1'][...,::-1], # flip color channel
        }
        return ob

class ResetFreeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.task_id = 0 # 0: place into silver bin, 1: place into brown bin

    def reset(self, task_id=0):
        self.task_id = task_id
        print(f'reset to task {self.task_id}')
        if self.task_id == 0:
            self.resetpos[1] = self.centerpos[1] + 0.1
        else:
            self.resetpos[1] = self.centerpos[1] - 0.1
        return self.env.reset()