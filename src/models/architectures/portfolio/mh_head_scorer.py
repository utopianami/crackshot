"""Multi-head MLP portfolio scorer.

Same MLP body as ``CrossSectionalScorerMLP`` but with ``n_heads`` parallel
linear heads on the final hidden representation. The per-stock scalar score
is the mean of the head outputs. Each head has independent random init, so
they may specialize during training instead of collapsing to a single linear
projection.

Per-stock signature: ``model(x: [D], key, inference) -> scalar``.

The full ``[n_heads]`` vector is also accessible via ``head_scores(x, key)``
for callers that want to inspect or supervise per-head outputs (multi-task
loss, etc.). ``__call__`` itself returns the scalar mean, so this scorer is
drop-in compatible with the existing per-stock training step.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _act(name: str):
    name = name.lower()
    return {"gelu": jax.nn.gelu, "relu": jax.nn.relu, "silu": jax.nn.silu}[name]


class MHHeadScorer(eqx.Module):
    layers: list
    norms: list
    drops: list
    heads: list  # list of eqx.nn.Linear(prev, 1)
    n_heads: int = eqx.field(static=True)
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden_units: list[int],
        dropout: float,
        n_heads: int,
        use_layernorm: bool,
        activation: str,
        *,
        key: jax.Array,
    ):
        n_body = len(hidden_units)
        keys = jax.random.split(key, n_body + n_heads)
        layers, norms, drops = [], [], []
        prev = in_features
        for i, h in enumerate(hidden_units):
            layers.append(eqx.nn.Linear(prev, h, key=keys[i]))
            norms.append(eqx.nn.LayerNorm(h) if use_layernorm else None)
            drops.append(eqx.nn.Dropout(p=dropout))
            prev = h
        self.layers = layers
        self.norms = norms
        self.drops = drops
        self.heads = [eqx.nn.Linear(prev, 1, key=keys[n_body + i]) for i in range(n_heads)]
        self.n_heads = n_heads
        self.activation = activation

    def _body(self, x: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        act = _act(self.activation)
        keys = jax.random.split(key, len(self.layers))
        h = x
        for layer, norm, drop, sk in zip(self.layers, self.norms, self.drops, keys):
            h = layer(h)
            if norm is not None:
                h = norm(h)
            h = act(h)
            h = drop(h, key=sk, inference=inference)
        return h

    def head_scores(self, x: jax.Array, *, key: jax.Array, inference: bool = True) -> jax.Array:
        """Return per-head scalars [n_heads]."""
        h = self._body(x, key=key, inference=inference)
        outs = jnp.stack([head(h).squeeze(-1) for head in self.heads])  # [n_heads]
        return outs

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool = False) -> jax.Array:
        outs = self.head_scores(x, key=key, inference=inference)
        return jnp.mean(outs)
