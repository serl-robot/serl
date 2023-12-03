"""Implementations of algorithms for continuous control."""
from functools import partial
from itertools import zip_longest
from typing import Callable, Optional, Sequence, Tuple

import gymnasium as gym
import jax
import optax
from flax import struct
from flax.training.train_state import TrainState

from serl.utils.augmentations import batched_random_crop
from serl.agents.sac.sac_learner import SACLearner
from serl.agents.sac.temperature import Temperature
from serl.data.dataset import DatasetDict
from serl.distributions import TanhNormal
from serl.networks import MLP, Ensemble, PixelMultiplexer, StateActionValue
from serl.networks.encoders import D4PGEncoder, ResNetV2Encoder, MobileNetEncoder
from serl.utils.commons import _unpack, _share_encoder


class DrQLearner(SACLearner):
    data_augmentation_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        encoder: str = "d4pg",
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if encoder == "d4pg":
            encoder_cls = partial(
                D4PGEncoder,
                features=cnn_features,
                filters=cnn_filters,
                strides=cnn_strides,
                padding=cnn_padding,
            )
        elif encoder == "resnet":
            encoder_cls = partial(ResNetV2Encoder, stage_sizes=(2, 2, 2, 2))
        elif encoder == "mobilenet":
            from jeffnet.linen import create_model, EfficientNet
            MobileNet, mobilenet_variables = create_model('tf_mobilenetv3_large_100', pretrained=True)
            encoder_cls = partial(MobileNetEncoder, mobilenet=MobileNet, params=mobilenet_variables)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_cls = partial(TanhNormal, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=actor_cls,
            latent_dim=latent_dim,
            stop_gradient=True,
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
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
        critic_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=critic_cls,
            latent_dim=latent_dim,
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

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
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
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            data_augmentation_fn=data_augmentation_fn,
        )

    @partial(jax.jit, static_argnames=("utd_ratio", "pixel_keys"))
    def update(self, batch: DatasetDict, utd_ratio: int, pixel_keys=("pixels",)):
        '''
        Update the agent's parameters (actor and critic) using the batch of data from the replay buffer.
        We apply data augmentation to both observations and next_observation,
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

        rng, key = jax.random.split(new_agent.rng)
        observations = self.data_augmentation_fn(key, batch["observations"])
        rng, key = jax.random.split(rng)
        next_observations = self.data_augmentation_fn(key, batch["next_observations"])

        batch = batch.copy(
            add_or_replace={
                "observations": observations,
                "next_observations": next_observations,
            }
        )
        new_agent = new_agent.replace(rng=rng)
        return SACLearner.update(new_agent, batch, utd_ratio)
