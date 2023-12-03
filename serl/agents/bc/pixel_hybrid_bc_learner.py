"""Implementations of algorithms for hybrid continuous and discrete control."""

from functools import partial
from itertools import zip_longest
from typing import Callable, Dict, Optional, Sequence, Tuple, OrderedDict
from collections import OrderedDict
import gymnasium as gym
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
from serl.networks.encoders import MobileNetEncoder, D4PGEncoder
from serl.networks.one_d_output import OneDimOutput
from serl.agents.agent import Agent
from serl.agents import PixelBCLearner
from serl.utils.commons import _unpack


class PixelHybridBCLearner(PixelBCLearner):
    discrete_actor: TrainState

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
        It supports training a hybrid policy with both discrete and continous actions policies.
        """

        action_dim = action_space.shape[-1] # take out the last action dim, which is the gripper's discrete action
        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, discrete_actor_key = jax.random.split(rng, 3)

        if encoder == "d4pg":
            encoder_cls = partial(
                D4PGEncoder,
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

        discrete_actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        discrete_actor_cls = partial(OneDimOutput, base_cls=discrete_actor_base_cls)
        discrete_actor_def = PixelMultiplexer(
            encoder_cls=encoder_cls,
            network_cls=discrete_actor_cls,
            latent_dim=latent_dim,
            stop_gradient=True, # do not update the encoder params
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        discrete_actor_params = discrete_actor_def.init(discrete_actor_key, observations)["params"]
        discrete_actor = TrainState.create(
            apply_fn=discrete_actor_def.apply,
            params=discrete_actor_params,
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
            discrete_actor=discrete_actor,
            data_augmentation_fn=data_augmentation_fn,
        )

    @jax.jit
    def compute_actions(self, observations) -> jnp.ndarray:
        action = self.continuous_actor.apply_fn({"params": self.continuous_actor.params}, observations).mode()
        gripper_action = self.discrete_actor.apply_fn({"params": self.discrete_actor.params}, observations)[..., None]
        action = jnp.concatenate((action, gripper_action), axis=-1)
        return action

    def eval_actions(self, observations) -> np.ndarray:
        action = np.array(self.compute_actions(observations), copy=True)
        action[-1] = 0.5 if action[-1] > 0.5 else -0.5
        return action

    @partial(jax.jit, static_argnames=("utd_ratio", "pixel_keys"))
    def update(self, batch: DatasetDict, utd_ratio: int, pixel_keys=("pixels",)):
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

        rng, key1, key2 = jax.random.split(self.rng, 3)

        def loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = new_agent.actor.apply_fn(
                {"params": actor_params},
                batch["observations"],
                training=True,
                rngs={"dropout": key1},
            )
            nll = -dist.log_prob(batch["actions"][..., :-1]).mean() # only compute BC loss for the continous action dims
            actor_loss = nll
            return actor_loss, {"nll": nll}

        def gripper_loss_fn(gripper_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            logits = new_agent.discrete_actor.apply_fn(
                {"params": gripper_params},
                batch["observations"],
                training=True,
                rngs={"dropout": key2},
            )
            # compute bce loss for the last action dim, which is the gripper and discrete
            bce = optax.sigmoid_binary_cross_entropy(logits, (batch["actions"][..., -1:] > 0).squeeze()).mean()
            return bce, {"bce": bce}

        grads, info = jax.grad(loss_fn, has_aux=True)(new_agent.actor.params)
        new_actor = new_agent.actor.apply_gradients(grads=grads)

        griipper_grads, gripper_info = jax.grad(gripper_loss_fn, has_aux=True)(new_agent.discrete_actor.params)
        new_discrete_actor = new_agent.discrete_actor.apply_gradients(grads=griipper_grads)
        info.update(gripper_info)

        return self.replace(actor=new_actor, discrete_actor=new_discrete_actor, rng=rng), info
