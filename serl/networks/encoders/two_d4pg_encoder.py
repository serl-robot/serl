from typing import Sequence, Callable
import flax.linen as nn
import jax.numpy as jnp

from serl.networks import default_init
from serl.networks.spatial import SpatialLearnedEmbeddings

class TwoD4PGEncoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    filters: Sequence[int] = (2, 1, 1, 1)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training=False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        processed_tensors = []
        reshape = False

        # Loop through all the tensors in the input FrozenDict
        for key, tensor in x.items():
            # Expand dimensions if they are 3
            if tensor.ndim == 3:
                tensor = tensor[None, ...]
                reshape = True

            # Apply Conv layers
            for features, filter_, stride in zip(self.features, self.filters, self.strides):
                tensor = nn.Conv(
                    features,
                    kernel_size=(filter_, filter_),
                    strides=(stride, stride),
                    kernel_init=default_init(),
                    padding=self.padding,
                )(tensor)
                tensor = self.activations(tensor)

            tensor = SpatialLearnedEmbeddings(*(tensor.shape[1:]), 8)(tensor)
            processed_tensors.append(tensor)

        # Concatenate all processed tensors along the last axis
        concatenated_tensor = jnp.concatenate(processed_tensors, axis=-1)

        # Reshape if original tensors were 3D
        if reshape:
            concatenated_tensor = concatenated_tensor.reshape(-1)

        return concatenated_tensor
