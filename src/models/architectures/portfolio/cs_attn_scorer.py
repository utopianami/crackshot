"""Cross-stock attention portfolio scorer — MASTER-style (arxiv 2312.15235).

Core contribution (Li et al., AAAI 2024):
  Within a single timestamp, entities (stocks) attend to one another via
  multi-head self-attention over the symbol axis. This lets each stock's
  representation depend on the whole cross-section (e.g. relative ranks,
  sector-level flows) rather than just its own feature vector. A MLP head
  then emits a scalar score per stock.

Cross-entity signature (``cross_entity_scoring = True``):
  ``model(x: [S, D], key, inference) -> [S]``
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class _CrossEntityBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    ffn1: eqx.nn.Linear
    ffn2: eqx.nn.Linear
    drop: eqx.nn.Dropout

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads, query_size=d_model, key=k1,
        )
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.ffn1 = eqx.nn.Linear(d_model, d_model * ffn_mult, key=k2)
        self.ffn2 = eqx.nn.Linear(d_model * ffn_mult, d_model, key=k3)
        self.drop = eqx.nn.Dropout(dropout)

    def __call__(self, tokens: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        """tokens: [S, d_model] → [S, d_model]. Attention ACROSS the S axis."""
        k1, k2 = jax.random.split(key, 2)
        h = jax.vmap(self.norm1)(tokens)
        h = self.attn(h, h, h)
        h = self.drop(h, key=k1, inference=inference)
        tokens = tokens + h
        h = jax.vmap(self.norm2)(tokens)
        h = jax.vmap(self.ffn1)(h)
        h = jax.nn.gelu(h)
        h = jax.vmap(self.ffn2)(h)
        h = self.drop(h, key=k2, inference=inference)
        return tokens + h


class CSAttnScorer(eqx.Module):
    input_proj: eqx.nn.Linear
    blocks: list
    final_norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    cross_entity_scoring: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_mult: int,
        dropout: float,
        *,
        key: jax.Array,
    ):
        k_in, k_head, *k_blocks = jax.random.split(key, 2 + n_layers)
        self.input_proj = eqx.nn.Linear(in_features, d_model, key=k_in)
        self.blocks = [
            _CrossEntityBlock(d_model, n_heads, ffn_mult, dropout, key=k_blocks[i])
            for i in range(n_layers)
        ]
        self.final_norm = eqx.nn.LayerNorm(d_model)
        self.head = eqx.nn.Linear(d_model, 1, key=k_head)
        self.cross_entity_scoring = True

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        """x: [S, D] → scores [S]."""
        tokens = jax.vmap(self.input_proj)(x)                    # [S, d_model]
        keys = jax.random.split(key, len(self.blocks))
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens, key=keys[i], inference=inference)
        tokens = jax.vmap(self.final_norm)(tokens)
        scores = jax.vmap(self.head)(tokens)[:, 0]               # [S]
        return scores
