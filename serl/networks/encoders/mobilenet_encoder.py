from typing import Sequence, Callable
import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
import numpy as np
import jax


class MobileNetEncoder(nn.Module):
    mobilenet: Callable[..., Callable]
    params: FrozenDict
    stop_gradient: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training=False, divide_by=False, reshape=False) -> jnp.ndarray:
        '''
        encode an image using the mobilenet encoder
        TODO: it should work for all pretrained encoders, not just mobilenet.

        :param x: input image
        :param training: whether the network is in training mode
        :param divide_by: whether to divide the image by 255
        :param reshape: whether to reshape the image before passing into encoder
        :return: the encoded image
        '''

        mean = jnp.array((0.485, 0.456, 0.406))[None, ...]
        std = jnp.array((0.229, 0.224, 0.225))[None, ...]

        if reshape:
            x = jnp.reshape(x, (*x.shape[:-2], -1))

        if divide_by:
            x = x.astype(jnp.float32) / 255.0
            x = (x - mean) / std

        if x.ndim == 3:
            x = x[None, ...]
            x = self.mobilenet.apply(self.params, x, mutable=False, training=False)
        elif x.ndim == 4:
            x = self.mobilenet.apply(self.params, x, mutable=False, training=False)
        else:
            raise NotImplementedError('ndim is not 3 or 4')

        if self.stop_gradient:
            x = jax.lax.stop_gradient(x)

        return x
