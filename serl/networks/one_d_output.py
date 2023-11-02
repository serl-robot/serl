import flax.linen as nn
import jax.numpy as jnp

from serl.networks import default_init
from serl.networks.spectral import SpectralNormalization

class OneDimOutput(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        if self.base_cls:
            outputs = self.base_cls()(observations, *args, **kwargs)
        else:
            outputs = observations

        value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)
