"""Implementations of algorithms for continuous control."""

from functools import partial
from itertools import zip_longest
from typing import Callable, Dict, Optional, Sequence, Tuple, OrderedDict
from collections import OrderedDict
import gym
import jax
from jax import numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.core import FrozenDict, freeze
from flax.training.train_state import TrainState
import flax.linen as nn
from serl.utils.augmentations import batched_random_crop
from serl.data.dataset import DatasetDict
from serl.distributions import TanhNormal
from serl.networks import MLP, PixelMultiplexer
from serl.networks.encoders import TwoMobileNetEncoder, TwoD4PGEncoder
from serl.networks.one_d_output import OneDimOutput
from serl.agents.agent import Agent
from serl.utils.commons import _unpack


class PixelBCLearner(Agent):
    actor: TrainState

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        encoder: str = "d4pg",
        hidden_dims: Sequence[int] = (256, 256),
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
    ):
        """
        An implementation of pixel input based Behavioral Cloning.
        It assumes continous action space.
        """

        action_dim = action_space.shape
        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)

        if encoder == "d4pg":
            encoder_cls = partial(
                TwoD4PGEncoder,
                features=cnn_features,
                filters=cnn_filters,
                strides=cnn_strides,
                padding=cnn_padding,
            )
        elif encoder == "resnet":
            raise NotImplementedError
        elif encoder == "mobilenet":
            from jeffnet.linen import create_model, EfficientNet
            MobileNet, mobilenet_variables = create_model('tf_mobilenetv3_large_100', pretrained=True)
            encoder_cls = partial(TwoMobileNetEncoder, mobilenet=MobileNet, params=mobilenet_variables)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_cls = partial(TanhNormal, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=actor_cls,
            latent_dim=latent_dim,
            stop_gradient=False, # do not update the encoder params
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
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
            data_augmentation_fn=data_augmentation_fn,
        )

    @partial(jax.jit, static_argnames=("pixel_keys",))
    def update(self, batch: DatasetDict, pixel_keys=("pixels",)):
        new_agent = self

        if "pixels" not in batch["next_observations"]:
            batch = _unpack(batch)

        key, rng = jax.random.split(new_agent.rng)
        observations = self.data_augmentation_fn(key, batch["observations"])

        eps = 1e-5
        actions = batch["actions"]
        batch = batch.copy(
            add_or_replace={
                "observations": observations,
                "actions": jnp.clip(actions, -1 + eps, 1 - eps)
            }
        )

        rng, key1 = jax.random.split(self.rng, 3)

        def loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = new_agent.actor.apply_fn(
                {"params": actor_params},
                batch["observations"],
                training=True,
                rngs={"dropout": key1},
            )
            nll = -dist.log_prob(batch["actions"]).mean()
            actor_loss = nll
            return actor_loss, {"nll": nll}

        grads, info = jax.grad(loss_fn, has_aux=True)(new_agent.actor.params)
        new_actor = new_agent.actor.apply_gradients(grads=grads)

        return self.replace(actor=new_actor, rng=rng), info
