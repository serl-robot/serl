import flax.linen as nn
import jax.numpy as jnp

from serl.networks import default_init


class NDimOutput(nn.Module):
    n_dim: int
    base_cls: nn.Module
    spectral_norm: bool = False

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        if self.base_cls:
            outputs = self.base_cls()(observations, *args, **kwargs)
        else:
            outputs = observations

        value = nn.Dense(self.n_dim, kernel_init=default_init())(outputs)
        return value
