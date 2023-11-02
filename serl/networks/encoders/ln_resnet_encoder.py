# Based on:
# https://github.com/google/flax/blob/main/examples/imagenet/models.py
# and
# https://github.com/google-research/big_transfer/blob/master/bit_jax/models.py
from functools import partial
from typing import Any, Callable, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp
from flax import linen as nn

from serl.networks import default_init


class ResNetV2Block(nn.Module):
    """ResNet block."""

    filters: int
    conv_cls: Type[nn.Module]
    norm_cls: Type[nn.Module]
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.norm_cls()(x)
        y = self.act(y)
        y = self.conv_cls(self.filters, (3, 3), self.strides)(y)
        y = self.norm_cls()(y)
        y = self.act(y)
        y = self.conv_cls(self.filters, (3, 3))(y)

        if residual.shape != y.shape:
            residual = self.conv_cls(self.filters, (1, 1), self.strides)(residual)

        return residual + y


class MyGroupNorm(nn.GroupNorm):
    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)

class ResNetV2Encoder(nn.Module):
    """ResNetV2."""

    stage_sizes: Tuple[int]
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        if x.ndim == 3:
            x_0 = x[:x.shape[0] // 2]
            x_1 = x[x.shape[0] // 2:]
        elif x.ndim == 4:
            x_0 = x[:, :x.shape[1] // 2, ...]
            x_1 = x[:, x.shape[1] // 2:, ...]
        conv_cls = partial(
            nn.Conv, use_bias=False, dtype=self.dtype, kernel_init=default_init()
        )
        norm_cls = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)

        if x.shape[-2] == 224:
            x = conv_cls(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)])(x)
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        else:
            x = conv_cls(self.num_filters, (3, 3))(x)

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = ResNetV2Block(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv_cls=conv_cls,
                    norm_cls=norm_cls,
                    act=self.act,
                )(x)

        x = norm_cls()(x)
        x = self.act(x)
        return x.reshape((*x.shape[:-3], -1))

class TwoResNetV2Encoder(nn.Module):
    """ResNetV2."""

    stage_sizes: Tuple[int]
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        if x.ndim == 3:
            x_0 = x[:x.shape[0] // 2]
            x_1 = x[x.shape[0] // 2:]
        elif x.ndim == 4:
            x_0 = x[:, :x.shape[1] // 2, ...]
            x_1 = x[:, x.shape[1] // 2:, ...]

        conv_cls = partial(
            nn.Conv, use_bias=False, dtype=self.dtype, kernel_init=default_init()
        )
        norm_cls = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        if x_0.shape[-2] == 224:
            x_0 = conv_cls(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)])(x_0)
            x_0 = nn.max_pool(x_0, (3, 3), strides=(2, 2), padding="SAME")
            x_1 = conv_cls(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)])(x_1)
            x_1 = nn.max_pool(x_1, (3, 3), strides=(2, 2), padding="SAME")
        else:
            x_0 = conv_cls(self.num_filters, (3, 3))(x_0)
            x_1 = conv_cls(self.num_filters, (3, 3))(x_1)

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x_0 = ResNetV2Block(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv_cls=conv_cls,
                    norm_cls=norm_cls,
                    act=self.act,
                )(x_0)
                x_1 = ResNetV2Block(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv_cls=conv_cls,
                    norm_cls=norm_cls,
                    act=self.act,
                )(x_1)

        x_0 = norm_cls()(x_0)
        x_0 = self.act(x_0)
        x_1 = norm_cls()(x_1)
        x_1 = self.act(x_1)
        x_0 = x_0.reshape((*x_0.shape[:-3], -1))
        x_1 = x_1.reshape((*x_1.shape[:-3], -1))
        return x_0, x_1
        