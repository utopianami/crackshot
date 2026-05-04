"""DCN-V2 portfolio scorer (Wang et al. 2021, arxiv 2008.13535).

Cross network learns explicit feature interactions:
    x_{l+1} = x_0 * (W x_l + b) + x_l
where * is elementwise multiply. After K cross layers, an MLP head produces
a scalar score per stock.

Per-stock signature: ``model(x: [D], key, inference) -> scalar``.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _act(name: str):
    name = name.lower()
    return {"gelu": jax.nn.gelu, "relu": jax.nn.relu, "silu": jax.nn.silu}[name]


class _CrossLayer(eqx.Module):
    """x_{l+1} = x_0 ⊙ (W x_l + b) + x_l. W is [D, D]."""
    lin: eqx.nn.Linear

    def __init__(self, dim: int, *, key: jax.Array):
        self.lin = eqx.nn.Linear(dim, dim, key=key)

    def __call__(self, x_l: jax.Array, x_0: jax.Array) -> jax.Array:
        return x_0 * self.lin(x_l) + x_l


class DCNScorer(eqx.Module):
    in_proj: eqx.nn.Linear
    in_norm: eqx.nn.LayerNorm | None
    cross_layers: list
    mlp_layers: list
    mlp_norms: list
    mlp_drops: list
    head: eqx.nn.Linear
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden_units: list[int],
        n_cross_layers: int,
        dropout: float,
        activation: str,
        use_layernorm: bool,
        *,
        key: jax.Array,
    ):
        width = int(hidden_units[0])
        n_mlp = len(hidden_units)
        n_keys = 1 + n_cross_layers + n_mlp + 1
        keys = jax.random.split(key, n_keys)
        ki = 0

        self.in_proj = eqx.nn.Linear(in_features, width, key=keys[ki]); ki += 1
        self.in_norm = eqx.nn.LayerNorm(width) if use_layernorm else None

        self.cross_layers = []
        for _ in range(n_cross_layers):
            self.cross_layers.append(_CrossLayer(width, key=keys[ki])); ki += 1

        self.mlp_layers = []
        self.mlp_norms = []
        self.mlp_drops = []
        prev = width
        for h in hidden_units:
            self.mlp_layers.append(eqx.nn.Linear(prev, h, key=keys[ki])); ki += 1
            self.mlp_norms.append(eqx.nn.LayerNorm(h) if use_layernorm else None)
            self.mlp_drops.append(eqx.nn.Dropout(p=dropout))
            prev = h

        self.head = eqx.nn.Linear(prev, 1, key=keys[ki])
        self.activation = activation

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        act = _act(self.activation)
        x_0 = self.in_proj(x)
        if self.in_norm is not None:
            x_0 = self.in_norm(x_0)
        x_l = x_0
        for cl in self.cross_layers:
            x_l = cl(x_l, x_0)
        h = x_l
        keys = jax.random.split(key, len(self.mlp_layers))
        for layer, norm, drop, sk in zip(self.mlp_layers, self.mlp_norms, self.mlp_drops, keys):
            h = layer(h)
            if norm is not None:
                h = norm(h)
            h = act(h)
            h = drop(h, key=sk, inference=inference)
        return self.head(h).squeeze(-1)
