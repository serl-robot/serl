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
        x = []

        for image_key in self.pixel_keys:
            # divide_by will convert the image to float32 and divide by 255
            # reshape will merge the stacking dimension with the channel dimension
            encoded_image = self.encoder_cls(name=f"{image_key}_encoder")(
                image_obs[image_key], training, divide_by=True, reshape=True
            )
            # We do not update conv or spatial embedding layers with policy gradients.
            if self.stop_gradient:
                encoded_image = jax.lax.stop_gradient(encoded_image)
            encoded_image = nn.Dense(256, kernel_init=default_init())(encoded_image)
            encoded_image = nn.LayerNorm()(encoded_image)
            encoded_image = nn.tanh(encoded_image)
            x.append(encoded_image)

        x = jnp.concatenate(x, axis=-1)

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
