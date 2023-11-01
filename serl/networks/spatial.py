from typing import Sequence, Callable
import flax.linen as nn
import jax.numpy as jnp

class SpatialLearnedEmbeddings(nn.Module):
    height: int
    width: int
    channel: int
    num_features: int = 5
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, features):
        """ 
        features is B x H x W X C
        """
        kernel = self.param('kernel',
                            nn.initializers.lecun_normal(),
                            (self.height, self.width, self.channel, self.num_features),
                            self.param_dtype)

        batch_size = features.shape[0]
        # assert len(features.shape) == 4
        features = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2))
        features = jnp.reshape(features, [batch_size, -1])
        return features