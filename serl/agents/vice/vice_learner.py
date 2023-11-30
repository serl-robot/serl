"""Implementations of algorithms for continuous control."""

from functools import partial
from itertools import zip_longest
from typing import Callable, Dict, Optional, Sequence, Tuple, OrderedDict
from collections import OrderedDict
import gymnasium as gym
import jax
from jax import numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict, freeze
from flax.training.train_state import TrainState
import flax.linen as nn

from serl.utils.augmentations import batched_random_crop
from serl.agents.sac.sac_learner import SACLearner
from serl.agents.drq.drq_learner import DrQLearner
from serl.agents.sac.temperature import Temperature
from serl.data.dataset import DatasetDict
from serl.distributions import TanhNormal
from serl.networks import MLP, Ensemble, PixelMultiplexer, StateActionValue
from serl.networks.encoders import TwoMobileNetEncoder, MobileNetEncoder, TwoD4PGEncoder
from serl.networks.encoded_encoder import EncodedEncoder
from serl.networks.one_d_output import OneDimOutput
from serl.utils.commons import _unpack, _share_encoder


class VICELearner(DrQLearner):
    vice_classifiers: OrderedDict[str, TrainState]
    vice_label_smoothing: float
    vice_goal_pool: jnp.ndarray
    vice_encoder: TrainState
    vice_encoder_params: FrozenDict

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        vice_lr: float = 3e-4,
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
        vice_dropout_rate: Optional[float] = None,
        vice_label_smoothing: float = 0.1,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
        vice_goal_pool: jnp.ndarray = None
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key, vice_encoder_key = jax.random.split(rng, 5)
        rng_vice_keys = jax.random.split(rng, 1 + len(pixel_keys))
        rng, vice_keys = rng_vice_keys[0], rng_vice_keys[1:]

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
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
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

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        vice_base_cls = partial(
            MLP,
            activations=nn.leaky_relu, # use leaky relu for Gradient Penalty regularization
            hidden_dims=(256, ),
            activate_final=True,
            dropout_rate=vice_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        vice_cls = partial(OneDimOutput, base_cls=vice_base_cls)

        half_observation = observations.copy()
        vice_encoder_cls = MobileNetEncoder(MobileNet, mobilenet_variables, name="vice_encoder", stop_gradient=True)
        vice_encoder_params = vice_encoder_cls.init(
            vice_encoder_key,
            half_observation[pixel_keys[0]],
            reshape=True,
            divide_by=True)
        vice_encoder = TrainState.create(
            apply_fn=vice_encoder_cls.apply,
            params=vice_encoder_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        '''
        creating VICE reward classifier for each camera input
        '''
        vice_classifiers = OrderedDict()
        for p_key, r_key in zip(pixel_keys, vice_keys):
            half_observation[p_key] = vice_encoder.apply_fn(
                {'params': vice_encoder_params},
                observations[p_key],
                reshape=True,
                divide_by=True
            )
            vice_def = EncodedEncoder(
                network_cls=vice_cls,
                latent_dim=latent_dim,
                stop_gradient=False,
                pixel_key=p_key,
                dropout_rate=vice_dropout_rate,
            )
            vice_params = vice_def.init(r_key, half_observation)['params']
            vice = TrainState.create(
                apply_fn=vice_def.apply,
                params=vice_params,
                tx=optax.adam(learning_rate=vice_lr),
            )
            vice_classifiers[p_key] = vice

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
            vice_classifiers=vice_classifiers,
            vice_label_smoothing=vice_label_smoothing,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            data_augmentation_fn=data_augmentation_fn,
            vice_goal_pool=vice_goal_pool,
            vice_encoder=vice_encoder,
            vice_encoder_params=vice_encoder_params,
        )

    @jax.jit
    def encode(self, x):
        '''
        encode the input image using the vice encoder. In the paper, we use a frozen pretrained MobileNetV3

        :param x: input images
        :return: encoded embeddings
        '''
        return self.vice_encoder.apply_fn(
            {'params': self.vice_encoder_params},
            x,
            reshape=True,
            divide_by=True
        )

    @partial(jax.jit, static_argnames="pixel_keys")
    def vice_reward(self, observations: FrozenDict, pixel_keys: Tuple[str, ...] = ("pixels",)):
        '''
        compute the VICE reward for each camera input. In the paper, it uses the average of VICE rewards from all cameras

        :param batch: batch of transitions from the replay buffer
        :param pixel_keys: camera inputs keys
        :return: updated agent, VICE rewards
        '''
        rng = self.rng
        vice_rews = OrderedDict()

        for p_key in pixel_keys:
            vice_classifier = self.vice_classifiers[p_key]
            key, rng = jax.random.split(rng)
            compute_rews = lambda x: nn.sigmoid(vice_classifier.apply_fn(
                {'params': vice_classifier.params,},
                x,
                training=False,
                rngs={'dropout': key},
            ))
            half_observation = observations.copy(
                add_or_replace={
                    p_key: self.encode(observations[p_key])
                }
            )
            vice_rews[p_key] = compute_rews(half_observation)

        vice_rews['mean'] = jnp.mean(jnp.asarray([*vice_rews.values()]), axis=0)
        return self.replace(rng=rng), vice_rews

    @partial(jax.jit, static_argnames=("utd_ratio", "pixel_keys"))
    def update(self, batch: DatasetDict, utd_ratio: int, pixel_keys=("pixels",)):
        '''
        update the actor and critic using the VICE reward.

        :param batch: batch of transitions from the replay buffer
        :param utd_ratio: numbers of steps on the critics per update
        :param pixel_keys: camera inputs keys
        :return: updated agent, update info
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

        new_agent, all_vice_rews = new_agent.vice_reward(observations, pixel_keys)
        vice_rews = jnp.log(all_vice_rews['mean'] / (1-all_vice_rews['mean']))

        batch = batch.copy(
            add_or_replace={
                "observations": observations,
                "next_observations": next_observations,
                'rewards': vice_rews,
            }
        )
        agent, update_info = SACLearner.update(new_agent, batch, utd_ratio)
        update_info.update({
            "vice_rews": jnp.mean(vice_rews),
        })
        return agent, update_info

    @partial(jax.jit, static_argnames="pixel_keys")
    def update_classifier(self, batch: DatasetDict, pixel_keys=("pixels",)):
        '''
        update the VICE reward classifier using the BCE loss.
        addtional regularization techniques are also used: mixup, label smoothing, and gradient penalty regularization
        to prevent GAN mode collapse.

        :param batch: batch of transitions from the replay buffer, here we only use the images
        :param pixel_keys: camera inputs keys
        :return: updated agent, update info
        '''
        rng = self.rng
        new_vice_classifiers = OrderedDict()

        if pixel_keys[0] not in batch["next_observations"]:
            batch = _unpack(batch)

        def _sample(rng, x, batch_size: int):
            # helper function to sample positive images from the goal pool
            key, rng = jax.random.split(rng)
            indx = jax.random.randint(
                key, (batch_size,), minval=0, maxval=len(x)
            )
            return rng, jax.tree_map(
                lambda d: jnp.take(d, indx, axis=0), x
            )

        def mixup_data_rng(key_0, key_1, x: jnp.ndarray, y: jnp.ndarray, alpha=1):
            '''
            performs mixup regularization on the input images and labels

            :param key_0: random key to generate beta distribution
            :param key_1: random key for mixup
            :param x: input images
            :param y: input labels
            :param alpha: mixup hyperparameter
            :return: mixed inputs, pairs of targets, and lambda
            '''

            if alpha > 0:
                lam = jax.random.beta(key_0, alpha, alpha)
            else:
                lam = 1
            batch_size = x.shape[0]
            index = jax.random.permutation(key_1, batch_size)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam

        observations = batch["observations"]
        key, rng = jax.random.split(rng)
        # data augmentation on images
        aug_observations = self.data_augmentation_fn(key, observations)
        all_info = {}
        for p_key in pixel_keys:
            vice_classifier = self.vice_classifiers[p_key]
            pixels = observations[p_key]
            batch_size = pixels.shape[0]

            key, rng = jax.random.split(rng)
            # sample positive images from the goal pool and create a dict of goal observations
            rng, goal_samples = _sample(rng, self.vice_goal_pool[p_key], batch_size=batch_size)
            goal_observations = observations.copy(
                add_or_replace={
                    p_key: goal_samples,
                }
            )

            # augment goal images
            key, rng = jax.random.split(rng)
            aug_goal_observations = self.data_augmentation_fn(key, goal_observations)

            pixels = observations[p_key]
            aug_pixels = aug_observations[p_key]
            goal_pixels = goal_observations[p_key]
            aug_goal_pixels = aug_goal_observations[p_key]

            # concatenate all images for update
            all_obs_pixels = jnp.concatenate([pixels, aug_pixels], axis=0)
            all_goal_pixels = jnp.concatenate([goal_pixels, aug_goal_pixels], axis=0)
            all_pixels = jnp.concatenate([all_goal_pixels, all_obs_pixels], axis=0)

            # create labels
            ones = jnp.ones((batch_size * 2, 1))
            zeros = jnp.zeros((batch_size * 2, 1))
            y_batch = jnp.concatenate([ones, zeros], axis=0)
            y_batch = y_batch.squeeze(-1)

            # label smoothing, might help with bad numerical issue with logits
            y_batch = y_batch * (1- 0.2) + 0.5 * 0.2

            # encode images into embeddings
            encoded = self.encode(all_pixels)

            # perform mixup
            key_0, key_1, rng = jax.random.split(rng, 3)            
            mix_encoded, y_a_0, y_b_0, lam_0 = mixup_data_rng(key_0, key_1, encoded, y_batch)
            mix_observations = observations.copy(
                add_or_replace = {
                    p_key: mix_encoded,
                }
            )

            # interpolate for Gradient Penalty regularization
            key, rng = jax.random.split(rng)
            '''
            generate random epsilon for each sample in the batch, the shape here depends on the shape of the encoded embeddings.
            Here are some examples:
            epsilon = jax.random.uniform(key, shape=(all_obs_pixels.shape[0], 1, 1, 1, 1))
            epsilon = jax.random.uniform(key, shape=(all_obs_pixels.shape[0], 1))
            '''
            epsilon = jax.random.uniform(key, shape=(all_obs_pixels.shape[0], 1, 1, 1))
            gp_encoded = epsilon * mix_encoded[:len(mix_encoded) // 2] + (1 - epsilon) * mix_encoded[len(mix_encoded) // 2:]
            gp_observations = observations.copy(
                add_or_replace={
                    p_key: gp_encoded,
                }
            )
            # remove all non pixel inputs keys from the batch
            remove_keys = [k for k in gp_observations.keys() if k != p_key]
            for k in remove_keys:
                gp_observations, _ = gp_observations.pop(k)

            key, rng = jax.random.split(rng, 2)
            def mixup_loss_fn(vice_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
                y_hat = vice_classifier.apply_fn(
                    {'params': vice_params,},
                    mix_observations,
                    training=True,
                    rngs={'dropout': key},
                )
                bce_loss_a = jnp.mean(optax.sigmoid_binary_cross_entropy(y_hat, y_a_0))
                bce_loss_b = jnp.mean(optax.sigmoid_binary_cross_entropy(y_hat, y_b_0))
                bce_loss = lam_0 * bce_loss_a + (1 - lam_0) * bce_loss_b
                return bce_loss, {f'bce_loss_{p_key}': bce_loss,}

            key, rng = jax.random.split(rng, 2)
            def gp_loss_fn(vice_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
                bce_loss, info = mixup_loss_fn(vice_params)
                helper_fn = lambda x: vice_classifier.apply_fn(
                    {'params': vice_params},
                    x,
                    training=True,
                    rngs={'dropout': key},
                )
                grad_wrt_input = jax.vmap(jax.grad(helper_fn), in_axes=0, out_axes=0)
                gradients = grad_wrt_input(gp_observations)
                gradients = jax.tree_util.tree_flatten(gradients)[0][0]
                gradients = gradients.reshape((gradients.shape[0], -1))
                grad_norm = jnp.sqrt(jnp.sum((gradients ** 2 + 1e-6), axis=1))
                grad_penalty = jnp.mean((grad_norm - 1) ** 2)
                return bce_loss + 10 * grad_penalty, {f'bce_loss_{p_key}': bce_loss ,f'grad_norm_{p_key}': grad_norm.mean()}

            grads, info = jax.grad(gp_loss_fn, has_aux=True)(vice_classifier.params)
            vice = vice_classifier.apply_gradients(grads=grads)
            new_vice_classifiers[p_key] = vice
            all_info.update(info)

        return self.replace(vice_classifiers=new_vice_classifiers, rng=rng), all_info
