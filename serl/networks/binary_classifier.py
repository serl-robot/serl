from flax import linen as nn
from typing import Dict, Optional, Tuple, Type, Union
from serl.networks.encoders import TwoMobileNetEncoder

class BinaryClassifier(nn.Module):
    hidden_dim: int
    encoder_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, x, training=False):
        x = self.encoder_cls(name=f"image_encoder")(x, training)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not training)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

