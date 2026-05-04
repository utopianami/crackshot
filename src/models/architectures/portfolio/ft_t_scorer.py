"""FT-Transformer portfolio scorer (arxiv 2106.11959, NeurIPS 2021).

Core contribution (Gorishniy et al.):
  Each numeric feature x_i is tokenized into an embedding ``t_i = x_i · W_i + b_i``
  (a *learned feature-wise* Linear — equivalent to NumericalFeatureTokenizer in
  the original paper). A learnable [CLS] token is prepended. Self-attention
  across feature tokens lets the model learn inter-feature interactions; the
  final [CLS] representation is projected to the prediction.

Per-stock signature: ``model(x: [D], key, inference) -> scalar``.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class _FeatureTokenizer(eqx.Module):
    """Numeric-only tokenizer: per-feature Linear W_i + b_i on scalar x_i."""
    W: jax.Array   # [n_features, d_token]
    b: jax.Array   # [n_features, d_token]

    def __init__(self, n_features: int, d_token: int, *, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.W = jax.random.normal(k1, (n_features, d_token)) * (1.0 / jnp.sqrt(d_token))
        self.b = jnp.zeros((n_features, d_token))

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: [n_features] → [n_features, d_token]"""
        return x[:, None] * self.W + self.b


class _TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    ffn1: eqx.nn.Linear
    ffn2: eqx.nn.Linear
    drop: eqx.nn.Dropout

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float, *, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads, query_size=d_model, key=k1,
        )
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.ffn1 = eqx.nn.Linear(d_model, d_model * ffn_mult, key=k2)
        self.ffn2 = eqx.nn.Linear(d_model * ffn_mult, d_model, key=k3)
        self.drop = eqx.nn.Dropout(dropout)

    def __call__(self, tokens: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        """tokens: [T, d_model] → [T, d_model]"""
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


class FTTransformerScorer(eqx.Module):
    tokenizer: _FeatureTokenizer
    cls_token: jax.Array       # [d_token]
    blocks: list
    final_norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    d_token: int = eqx.field(static=True)

    def __init__(
        self,
        n_features: int,
        d_token: int,
        n_layers: int,
        n_heads: int,
        ffn_mult: int,
        dropout: float,
        *,
        key: jax.Array,
    ):
        k_tok, k_cls, k_head, *k_blocks = jax.random.split(key, 3 + n_layers)
        self.tokenizer = _FeatureTokenizer(n_features, d_token, key=k_tok)
        self.cls_token = 0.02 * jax.random.normal(k_cls, (d_token,))
        self.blocks = [
            _TransformerBlock(d_token, n_heads, ffn_mult, dropout, key=k_blocks[i])
            for i in range(n_layers)
        ]
        self.final_norm = eqx.nn.LayerNorm(d_token)
        self.head = eqx.nn.Linear(d_token, 1, key=k_head)
        self.d_token = d_token

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        """x: [D] → scalar score."""
        tokens = self.tokenizer(x)                                # [D, d_token]
        tokens = jnp.concatenate([self.cls_token[None, :], tokens], axis=0)  # [D+1, d_token]
        keys = jax.random.split(key, len(self.blocks))
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens, key=keys[i], inference=inference)
        cls = self.final_norm(tokens[0])                         # [d_token]
        return self.head(cls)[0]
