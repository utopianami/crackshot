"""Cross-sectional MLP — per-stock MLP with market context features.

Input: [stock_features, market_mean, market_std, z_score] per stock.
Output: classification logits for each horizon.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


class CrossSectionalMLP(eqx.Module):
    """Simple MLP that takes cross-sectional augmented features.

    Input dimension: n_features * 3  (raw + market_mean + z_score)
    Output: [num_horizons, num_classes] logits
    """

    layers: list
    dropouts: list
    norms: list
    head: eqx.nn.Linear
    num_horizons: int
    num_classes: int
    use_layernorm: bool
    activation: str

    def __init__(
        self,
        in_features: int,
        hidden_units: list[int],
        dropout: float,
        num_classes: int,
        num_horizons: int,
        *,
        key: jax.Array,
        use_layernorm: bool = False,
        activation: str = "gelu",
    ):
        keys = jax.random.split(key, len(hidden_units) + 1)

        layers = []
        dropouts = []
        norms = []
        prev_dim = in_features
        for i, h in enumerate(hidden_units):
            layers.append(eqx.nn.Linear(prev_dim, h, key=keys[i]))
            dropouts.append(eqx.nn.Dropout(p=dropout))
            norms.append(eqx.nn.LayerNorm(h) if use_layernorm else None)
            prev_dim = h

        self.layers = layers
        self.dropouts = dropouts
        self.norms = norms
        self.head = eqx.nn.Linear(prev_dim, num_horizons * num_classes, key=keys[-1])
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        self.activation = activation

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool = False):
        """Forward pass.

        Args:
            x: [aug_features] augmented feature vector
            key: PRNG key for dropout
            inference: if True, disable dropout

        Returns:
            logits: [num_horizons, num_classes]
        """
        act_fn = {"gelu": jax.nn.gelu, "silu": jax.nn.silu, "relu": jax.nn.relu}[self.activation]
        for layer, drop, norm in zip(self.layers, self.dropouts, self.norms):
            x = layer(x)
            if norm is not None:
                x = norm(x)
            x = act_fn(x)
            key, subkey = jax.random.split(key)
            x = drop(x, key=subkey, inference=inference)

        logits = self.head(x)  # [H * C]
        return logits.reshape(self.num_horizons, self.num_classes)
