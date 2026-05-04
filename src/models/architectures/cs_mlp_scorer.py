"""Cross-sectional MLP scorer — scalar score per entity.

Same body as CrossSectionalMLP but the head emits a single scalar score
(no class logits). The portfolio head (SOFT long/short top-k) consumes
these scores to produce signed weights.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class CrossSectionalScorerMLP(eqx.Module):
    layers: list
    dropouts: list
    norms: list
    head: eqx.nn.Linear
    use_layernorm: bool
    activation: str

    def __init__(
        self,
        in_features: int,
        hidden_units: list[int],
        dropout: float,
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
        self.head = eqx.nn.Linear(prev_dim, 1, key=keys[-1])
        self.use_layernorm = use_layernorm
        self.activation = activation

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool = False) -> jax.Array:
        act_fn = {"gelu": jax.nn.gelu, "silu": jax.nn.silu, "relu": jax.nn.relu}[self.activation]
        h = x
        for layer, drop, norm in zip(self.layers, self.dropouts, self.norms):
            h = layer(h)
            if norm is not None:
                h = norm(h)
            h = act_fn(h)
            key, subkey = jax.random.split(key)
            h = drop(h, key=subkey, inference=inference)
        return self.head(h).squeeze(-1)  # scalar
