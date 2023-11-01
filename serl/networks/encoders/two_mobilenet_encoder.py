from typing import Sequence, Callable
import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
import jax

from serl.networks import default_init
from serl.networks.spatial import SpatialLearnedEmbeddings

class TwoMobileNetEncoder(nn.Module):
    mobilenet: nn.Module
    params: FrozenDict
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: FrozenDict[str, jnp.ndarray], training=False) -> jnp.ndarray:
        processed_tensors = []
        reshape = False
        mean = jnp.array((0.485, 0.456, 0.406))[None, ...]
        std = jnp.array((0.229, 0.224, 0.225))[None, ...]

        # Loop through all the tensors in the input FrozenDict
        for key, tensor in x.items():
            # Expand dimensions if they are 3
            if tensor.ndim == 3:
                tensor = tensor[None, ...]
                reshape = True

            # Apply mobilenet
            tensor = (tensor - mean) / std # normalize using ImageNet stats
            tensor = self.mobilenet.apply(self.params, tensor, training=False)
            # Apply SpatialLearnedEmbeddings and Dropout
            tensor = SpatialLearnedEmbeddings(*(tensor.shape[1:]), 8)(tensor)
            tensor = nn.Dropout(self.dropout_rate)(tensor, deterministic=not training)

            processed_tensors.append(tensor)

        # Concatenate all processed tensors along the last axis
        concatenated_tensor = jnp.concatenate(processed_tensors, axis=-1)

        # Reshape if original tensors were 3D
        if reshape:
            concatenated_tensor = concatenated_tensor.reshape(-1)

        return concatenated_tensor
