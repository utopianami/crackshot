"""Cross-sectional MLP — per-entity MLP with group context features.

Input: [entity_features, market_mean, market_std, z_score] per entity.
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
    gate_layers: list  # GLU gate projections (None when not using GLU)
    dropouts: list
    norms: list
    head: eqx.nn.Linear
    num_horizons: int
    num_classes: int
    use_layernorm: bool
    activation: str
    residual: bool
    use_glu: bool

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
        residual: bool = False,
        use_glu: bool = False,
    ):
        n_keys = len(hidden_units) * (2 if use_glu else 1) + 1
        keys = jax.random.split(key, n_keys)

        layers = []
        gate_layers = []
        dropouts = []
        norms = []
        prev_dim = in_features
        ki = 0
        for i, h in enumerate(hidden_units):
            layers.append(eqx.nn.Linear(prev_dim, h, key=keys[ki]))
            ki += 1
            if use_glu:
                gate_layers.append(eqx.nn.Linear(prev_dim, h, key=keys[ki]))
                ki += 1
            else:
                gate_layers.append(None)
            dropouts.append(eqx.nn.Dropout(p=dropout))
            norms.append(eqx.nn.LayerNorm(h) if use_layernorm else None)
            prev_dim = h

        self.layers = layers
        self.gate_layers = gate_layers
        self.dropouts = dropouts
        self.norms = norms
        self.head = eqx.nn.Linear(prev_dim, num_horizons * num_classes, key=keys[ki])
        self.num_horizons = num_horizons
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        self.activation = activation
        self.residual = residual
        self.use_glu = use_glu

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
        for layer, gate, drop, norm in zip(self.layers, self.gate_layers, self.dropouts, self.norms):
            if self.use_glu and gate is not None:
                h = layer(x) * jax.nn.sigmoid(gate(x))
            else:
                h = layer(x)
            if norm is not None:
                h = norm(h)
            if not (self.use_glu and gate is not None):
                h = act_fn(h)
            key, subkey = jax.random.split(key)
            h = drop(h, key=subkey, inference=inference)
            # Residual connection if dims match
            if self.residual and x.shape[-1] == h.shape[-1]:
                x = x + h
            else:
                x = h

        logits = self.head(x)  # [H * C]
        return logits.reshape(self.num_horizons, self.num_classes)
