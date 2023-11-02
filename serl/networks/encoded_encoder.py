from typing import Dict, Optional, Tuple, Type, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from serl.networks import default_init
from serl.networks.spatial import SpatialLearnedEmbeddings


class EncodedEncoder(nn.Module):
    network_cls: Type[nn.Module]
    latent_dim: int
    stop_gradient: bool = False
    pixel_key: str = "pixels"
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        training: bool = False,
    ) -> jnp.ndarray:
        observations = FrozenDict(observations)
        x = observations[self.pixel_key]

        if x.ndim == 3:
            x = x[None, :]

        x = SpatialLearnedEmbeddings(*(x.shape[1:]), 8)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)

        if x.shape[0] == 1:
            x = x.reshape(-1)
        else:
            x = x.reshape((x.shape[0], -1))

        if self.stop_gradient:
            # We do not update conv layers with policy gradients.
            x = jax.lax.stop_gradient(x)

        x = nn.Dense(512, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return self.network_cls()(x, training)
