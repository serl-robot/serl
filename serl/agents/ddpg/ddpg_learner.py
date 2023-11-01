"""Implementations of algorithms for continuous control."""
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from serl.agents.agent import Agent
from serl.data.dataset import DatasetDict
from serl.distributions import TanhNormal
from serl.networks import MLP, Ensemble, StateActionValue, subsample_ensemble

class DDPGLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    tau: float
    discount: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
    ):
        """
        An implementation of DDPG
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_def = TanhNormal(actor_base_cls, action_dim)
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
        critic_def = Ensemble(critic_cls, num=1)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )

        target_critic_def = Ensemble(critic_cls, num=1)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            tau=tau,
            discount=discount,
        )

    @jax.jit
    def compute_actions(self, observations:dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        sample actions from the actor with a small amount of noise injected
        TODO: make the noise scale configurable

        :param observations: a batch of observations
        :return: actions and jax random key
        '''
        key, rng = jax.random.split(self.rng)
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        actions = dist.sample(seed=key)
        key, rng = jax.random.split(self.rng)
        actions += jax.random.normal(key, shape=actions.shape) * 0.05
        return actions, rng

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        '''
        sample actions from the actor with a small amount of noise injected, and convert to numpy array on CPU
        update agent's rng

        :param observations: a batch of observations
        :return: actions in numpy array and jax random key
        '''
        actions, rng = self.compute_actions(observations)
        return np.asarray(actions), self.replace(rng=rng)

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        '''
        update DDPG actor

        :param observations: a batch of observations
        :return: updated agent and info dict
        '''
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            actor_loss = (-q).mean()
            return actor_loss, {"actor_loss": actor_loss}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        '''
        update DDPG critic

        :param observations: a batch of observations
        :return: updated agent and info dict
        '''
        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # Used only if there is an ensemble of Qs, which is not the case in DDPG.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, 1, 1
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_qs

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        '''
        update DDPG actor and critic updates

        :param observations: a batch of observations
        :param utd_ratio: number of critic updates per actor update
        :return: updated agent and info dict
        '''

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)

        return new_agent, {**actor_info, **critic_info}
