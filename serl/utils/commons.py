from serl.data.dataset import DatasetDict
import os
from flax.training import checkpoints
import numpy as np


def _unpack(batch: DatasetDict):
    '''
    Helps to minimize CPU to GPU transfer.
    Assuming that if next_observation is missing, it's combined with observation:

    :param batch: a batch of data from the replay buffer, a dataset dict
    :return: a batch of unpacked data, a dataset dict
    '''

    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][..., :-1]
            next_obs_pixels = batch["observations"][pixel_key][..., 1:]

            obs = batch["observations"].copy(add_or_replace={pixel_key: obs_pixels})
            next_obs = batch["next_observations"].copy(
                add_or_replace={pixel_key: next_obs_pixels}
            )
            batch = batch.copy(
                add_or_replace={"observations": obs, "next_observations": next_obs}
            )

    return batch


def _share_encoder(source, target):
    '''
    Share encoder params between source and target:
    
    :param source: the source network, TrainState
    :param target: the target network, TrainState
    '''

    replacers = {}
    for k, v in source.params.items():
        if "encoder" in k:
            replacers[k] = v

    # e.g., Use critic conv layers in actor:
    new_params = target.params.copy(add_or_replace=replacers)
    return target.replace(params=new_params)

def get_data(data, start, end):
    '''
    recursive helper function to extract data range from a dataset dict, which is used in the replay buffer

    :param data: the input data, can be a dataset dict or a numpy array from the replay buffer
    :param start: the start index of the data range
    :param end: the end index of the data range\
    :return: eventually returns the numpy array within range
    '''

    if type(data) == dict:
        return {k: get_data(v, start, end) for k,v in data.items()}
    return data[start:end]

def restore_checkpoint_(path, item, step):
    '''
    helper function to restore checkpoints from a path, checks if the path exists

    :param path: the path to the checkpoints folder
    :param item: the TrainState to restore
    :param step: the step to restore
    :return: the restored TrainState
    '''

    assert os.path.exists(path)
    return checkpoints.restore_checkpoint(path, item, step)

def _reset_weights(source, target):
    '''
    Reset weights of target to source
    TODO: change this to take params directly instead of TrainState
    :param source: the source network, TrainState
    :param target: the target network, TrainState
    '''

    replacers = {}
    for k, v in source.params.items():
        if "encoder" not in k:
            replacers[k] = v

    new_params = target.params.copy(add_or_replace=replacers)
    return target.replace(params=new_params)

def ema(series, alpha=0.5):
    '''
    Exponential moving average
    :param series: the input series
    :param alpha: the smoothing factor
    :return: the smoothed series
    '''

    smoothed = np.zeros_like(series, dtype=float)
    smoothed[0] = series[0]
    for i in range(1, len(series)):
        smoothed[i] = alpha * series[i] + (1-alpha) * smoothed[i-1]
    return smoothed

