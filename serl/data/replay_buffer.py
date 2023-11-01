import collections
from typing import Optional, Union

import gym
import gym.spaces
import jax
import numpy as np

from serl.data.dataset import Dataset, DatasetDict
from flax import jax_utils
from jax import numpy as jnp
from typing import Any, Iterator, Optional, Tuple, Sequence
from flax.core.frozen_dict import FrozenDict
import threading


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.
        queue = collections.deque()
        device = jax_utils._pmap_device_order()
        should_pmap = len(device) > 1

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data))
                # if should_pmap:
                #     data = [jax.tree_map(lambda d: d[i * len(d) // len(device): (i+1) * len(d) // len(device)], data) for i in range(len(device))]
                #     queue.append(jax.device_put_sharded(data, devices=device))
                # else:
                #     queue.append(jax.device_put(data))
                # data = [jax.tree_map(lambda d: d[i * len(d) // len(device): (i+1) * len(d) // len(device)], data) for i in range(len(device))]
                # queue.append(jax.device_put_sharded(data, devices=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices),indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(f'last_idx {last_idx} >= self._size {self._size}')
            last_idx, batch = self.download(last_idx, self._size)
            yield batch

class GPUReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        super().__init__(observation_space, action_space, capacity, next_observation_space)
        # Convert data to JAX arrays
        self.dataset_dict = jax.tree_map(jnp.asarray, self.dataset_dict)
        self.lock = threading.Lock()  # Lock for thread-safe updates

    def copy_arrays_recursively(self, src, start_idx, to_idx, key=None):
        if isinstance(src, np.ndarray):
            src = jax.device_put(src).block_until_ready()
            self.dataset_dict[key] = self.dataset_dict[key].at[start_idx: to_idx].set(src)
        elif isinstance(src, FrozenDict):
            for k in src.keys():
                self.copy_arrays_recursively(src[k], start_idx, to_idx, k)
        else:
            raise TypeError()

    def upload(self, data_dict: FrozenDict):
        with self.lock:
            start_idx = self._insert_index
            end_idx = start_idx + len(data_dict['observations'])
            self.copy_arrays_recursively(data_dict, start_idx, end_idx)
            self._insert_index = (self._insert_index + len(data_dict['observations'])) % self._capacity
            self._size = min(self._size + len(data_dict['observations']), self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        while True:
            indx_max, data = self.sample_jax(**sample_args)
            yield indx_max, data
