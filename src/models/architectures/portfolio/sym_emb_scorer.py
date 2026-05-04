"""Symbol-embedding portfolio scorer.

Concatenates a learnable per-symbol embedding to the augmented feature vector,
then runs the standard MLP body. Lets the model learn coin-specific biases
(e.g. BTC/ETH vs majors vs alt clusters) on top of the cross-sectional features.

Convention: the input vector ``x`` has shape ``[D+1]`` where ``x[-1]`` is the
symbol id encoded as a float (cast to int32 inside the call). Padded slots
must be marked with ``sym_id == n_symbols`` so they hit a dedicated padding
embedding row that the loss ignores via the existing ``mask_ts``.

Per-stock signature: ``model(x: [D+1], key, inference) -> scalar``.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _act(name: str):
    name = name.lower()
    return {"gelu": jax.nn.gelu, "relu": jax.nn.relu, "silu": jax.nn.silu}[name]


class SymEmbScorer(eqx.Module):
    embedding: jax.Array  # [n_symbols+1, emb_dim]; last row reserved for padding
    layers: list
    norms: list
    drops: list
    head: eqx.nn.Linear
    in_features: int = eqx.field(static=True)  # = D (augmented features only, excludes sym_id col)
    n_symbols: int = eqx.field(static=True)
    emb_dim: int = eqx.field(static=True)
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        n_symbols: int,
        emb_dim: int,
        hidden_units: list[int],
        dropout: float,
        use_layernorm: bool,
        activation: str,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, len(hidden_units) + 2)
        # Embedding matrix: n_symbols + 1 rows; last row = padding (initialized small)
        emb_key, *body_keys = keys
        self.embedding = 0.02 * jax.random.normal(emb_key, (n_symbols + 1, emb_dim))

        layers = []
        norms = []
        drops = []
        prev = in_features + emb_dim
        for i, h in enumerate(hidden_units):
            layers.append(eqx.nn.Linear(prev, h, key=body_keys[i]))
            norms.append(eqx.nn.LayerNorm(h) if use_layernorm else None)
            drops.append(eqx.nn.Dropout(p=dropout))
            prev = h
        self.layers = layers
        self.norms = norms
        self.drops = drops
        self.head = eqx.nn.Linear(prev, 1, key=body_keys[-1])
        self.in_features = in_features
        self.n_symbols = n_symbols
        self.emb_dim = emb_dim
        self.activation = activation

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        act = _act(self.activation)
        feat = x[: self.in_features]
        sid_f = x[self.in_features]
        # Clamp to valid range [0, n_symbols] (n_symbols = padding row).
        sid = jnp.clip(sid_f.astype(jnp.int32), 0, self.n_symbols)
        emb = self.embedding[sid]
        h = jnp.concatenate([feat, emb], axis=-1)
        keys = jax.random.split(key, len(self.layers))
        for layer, norm, drop, sk in zip(self.layers, self.norms, self.drops, keys):
            h = layer(h)
            if norm is not None:
                h = norm(h)
            h = act(h)
            h = drop(h, key=sk, inference=inference)
        return self.head(h).squeeze(-1)
