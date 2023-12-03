# !/usr/bin/env python3

# NOTE: this requires jaxrl_m to be installed: 
#       https://github.com/rail-berkeley/jaxrl_minimal

import gymnasium as gym
import jax
from jax import nn
from threading import Lock
from collections import deque
from functools import partial

from edgeml.trainer import TrainerConfig
from edgeml.data.data_store import DataStoreBase
from edgeml.trainer import TrainerConfig
from edgeml.data.serl_memory_efficient_replay_buffer import MemoryEfficientReplayBuffer

##############################################################################


class ReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        self.pixel_keys = [k for k in observation_space.spaces.keys() if k != 'state']
        MemoryEfficientReplayBuffer.__init__(self, observation_space, action_space, capacity, self.pixel_keys)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


##############################################################################


def make_agent(seed, sample_obs, sample_action):
    return SACAgent.create_states(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "softplus",
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=0.99,
        backup_entropy=True,
        critic_ensemble_size=10,
        critic_subsample_size=2,
    )

def make_trainer_config():
    return TrainerConfig(
        port_number=5488,
        broadcast_port=5489,
        request_types=["send-stats"]
    )


def make_wandb_logger(
    project: str = "edgeml",
    description: str = "jaxrl_m",
    debug: bool = False,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger
