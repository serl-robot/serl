from serl.data.dataset import DatasetDict

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