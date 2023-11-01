import functools
from typing import Optional, Type

import tensorflow_probability

from serl.distributions.tanh_transformed import TanhTransformedDistribution
import flax.linen as nn
import jax.numpy as jnp

from serl.networks import default_init


class Sigmoid(nn.Module):
    base_cls: Type[nn.Module]

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        outputs = self.base_cls()(observations, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)
        value = nn.sigmoid(value)
        return jnp.squeeze(value, -1)

