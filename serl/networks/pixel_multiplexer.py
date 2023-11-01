from typing import Dict, Optional, Tuple, Type, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from serl.networks import default_init

class PixelMultiplexer(nn.Module):
    encoder_cls: Type[nn.Module]
    network_cls: Type[nn.Module]
    latent_dim: int
    stop_gradient: bool = False
    pixel_keys: Tuple[str, ...] = ("pixels",)
    depth_keys: Tuple[str, ...] = ()

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        observations = FrozenDict(observations)
        image_obs, state_obs = observations.pop("state")
        reshape_img = lambda x: x.reshape(*x.shape[:-2], -1) / 255.0
        image_obs = jax.tree_map(reshape_img, image_obs)

        x = self.encoder_cls(name=f"image_encoder")(image_obs, training)
        if self.stop_gradient:
            # We do not update conv layers with policy gradients.
            x = jax.lax.stop_gradient(x)
        x = nn.Dense(512, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        if "state" in observations:
            y = nn.Dense(self.latent_dim, kernel_init=default_init())(
                observations["state"]
            )
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)

        if actions is None:
            return self.network_cls()(x, training)
        else:
            return self.network_cls()(x, actions, training)
