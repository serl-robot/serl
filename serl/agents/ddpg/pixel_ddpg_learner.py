"""Implementations of algorithms for continuous control."""

from functools import partial
from itertools import zip_longest
from typing import Callable, Dict, Optional, Sequence, Tuple, OrderedDict
from collections import OrderedDict
import gym
import jax
from jax import numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict, freeze
from flax.training.train_state import TrainState
import flax.linen as nn
# from flax.training import checkpoints
from serl.utils.augmentations import batched_random_crop
from serl.agents.ddpg.ddpg_learner import DDPGLearner
from serl.data.dataset import DatasetDict
from serl.distributions import TanhNormal
from serl.networks import MLP, Ensemble, PixelMultiplexer, StateActionValue
from serl.networks.encoders import TwoMobileNetEncoder, TwoD4PGEncoder

def _unpack(batch):
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


class PixelDDPGLearner(DDPGLearner):
    data_augmentation_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        encoder: str = "d4pg",
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
    ):
        """
        An implementation of pixel-based DDPG
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        if encoder == "d4pg":
            encoder_cls = partial(
                TwoD4PGEncoder,
                features=cnn_features,
                filters=cnn_filters,
                strides=cnn_strides,
                padding=cnn_padding,
            )
        elif encoder == "resnet":
            # TODO: option 1 refactor this to use ResNet from huggingface, option 2 use jax_resnet
            raise NotImplementedError
            from jax_resnet import pretrained_resnet, slice_variables
            ResNet, resnet_variables = pretrained_resnet(18)
            ResNet = ResNet()
            ResNet = nn.Sequential(ResNet.layers[0:3])
            resnet_variables = slice_variables(resnet_variables, end=3)
            encoder_cls = partial(TwoResNetEncoder, resnet=ResNet, params=resnet_variables)
        elif encoder == "mobilenet":
            # TODO: unfortunately, huggingface does not support many visual encoders in JAX, so we have to reply on https://github.com/Leo428/efficientnet-jax, forked from @rwightman
            from jeffnet.linen import create_model, EfficientNet
            MobileNet, mobilenet_variables = create_model('tf_mobilenetv3_large_100', pretrained=True)
            encoder_cls = partial(TwoMobileNetEncoder, mobilenet=MobileNet, params=mobilenet_variables)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_cls = partial(TanhNormal, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=actor_cls,
            latent_dim=latent_dim,
            stop_gradient=True, # do not update the encoder params
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=1)
        critic_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=critic_cls,
            latent_dim=latent_dim,
            stop_gradient=False,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        def data_augmentation_fn(rng, observations):
            for pixel_key, depth_key in zip_longest(pixel_keys, depth_keys):
                key, rng = jax.random.split(rng)
                observations = batched_random_crop(key, observations, pixel_key)
                if depth_key is not None:
                    observations = batched_random_crop(key, observations, depth_key)
            return observations

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            tau=tau,
            discount=discount,
            data_augmentation_fn=data_augmentation_fn
        )

    @partial(jax.jit, static_argnames=("utd_ratio", "pixel_keys"))
    def update(self, batch: DatasetDict, utd_ratio: int, pixel_keys=("pixels",)):
        '''
        Update the agent's parameters (actor and critic) using the batch of data from the replay buffer.
        The difference from DDPGLearner.update is that we apply data augmentation to both observations and next_observation,
        then we share the encoder params between actor and critic.

        :param batch: a batch of data from the replay buffer, a dataset dict
        :param utd_ratio: the number of times to update the critic for each update of the actor
        :param pixel_keys: pixel keys to apply data augmentation to
        :return: the updated agent and the update info dict
        '''

        new_agent = self

        if pixel_keys[0] not in batch["next_observations"]:
            batch = _unpack(batch)

        actor = _share_encoder(source=new_agent.critic, target=new_agent.actor)
        new_agent = new_agent.replace(actor=actor)

        key, rng = jax.random.split(new_agent.rng)
        observations = self.data_augmentation_fn(key, batch["observations"])
        key, rng = jax.random.split(rng)
        next_observations = self.data_augmentation_fn(key, batch["next_observations"])

        batch = batch.copy(
            add_or_replace={
                "observations": observations,
                "next_observations": next_observations,
            }
        )
        agent, update_info = DDPGLearner.update(new_agent, batch, utd_ratio)
        return agent, update_info

