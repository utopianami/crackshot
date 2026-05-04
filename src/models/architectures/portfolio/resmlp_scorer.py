"""ResMLP portfolio scorer (Touvron et al. 2021, arxiv 2105.03404).

Core contribution:
  Residual MLP blocks + LayerNorm let deep MLPs train stably without Transformer
  attention. Each block: x → LN → Linear → act → Linear → dropout → +x.
  We adapt the feed-forward part (skipping the cross-token affine of the
  original vision-ResMLP) since tabular inputs have no token sequence.

Per-stock signature: ``model(x: [D], key, inference) -> scalar``.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _act(name: str):
    name = name.lower()
    return {"gelu": jax.nn.gelu, "relu": jax.nn.relu, "silu": jax.nn.silu}[name]


class _ResBlock(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    norm: eqx.nn.LayerNorm | None
    drop: eqx.nn.Dropout
    activation: str = eqx.field(static=True)

    def __init__(self, dim: int, dropout: float, activation: str,
                 use_layernorm: bool, *, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.lin1 = eqx.nn.Linear(dim, dim, key=k1)
        self.lin2 = eqx.nn.Linear(dim, dim, key=k2)
        self.norm = eqx.nn.LayerNorm(dim) if use_layernorm else None
        self.drop = eqx.nn.Dropout(p=dropout)
        self.activation = activation

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        act = _act(self.activation)
        h = x
        if self.norm is not None:
            h = self.norm(h)
        h = self.lin1(h)
        h = act(h)
        h = self.lin2(h)
        h = self.drop(h, key=key, inference=inference)
        return x + h  # residual skip


class ResMLPScorer(eqx.Module):
    in_proj: eqx.nn.Linear
    blocks: list
    head_norm: eqx.nn.LayerNorm | None
    head: eqx.nn.Linear
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden_units: list[int],
        dropout: float,
        activation: str,
        use_layernorm: bool,
        *,
        key: jax.Array,
    ):
        # Use first hidden dim as the residual width; all blocks share it.
        width = int(hidden_units[0])
        n_blocks = len(hidden_units)
        keys = jax.random.split(key, n_blocks + 2)
        self.in_proj = eqx.nn.Linear(in_features, width, key=keys[0])
        self.blocks = [
            _ResBlock(width, dropout, activation, use_layernorm, key=keys[1 + i])
            for i in range(n_blocks)
        ]
        self.head_norm = eqx.nn.LayerNorm(width) if use_layernorm else None
        self.head = eqx.nn.Linear(width, 1, key=keys[-1])
        self.activation = activation

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        h = self.in_proj(x)
        keys = jax.random.split(key, len(self.blocks))
        for i, blk in enumerate(self.blocks):
            h = blk(h, key=keys[i], inference=inference)
        if self.head_norm is not None:
            h = self.head_norm(h)
        return self.head(h).squeeze(-1)
