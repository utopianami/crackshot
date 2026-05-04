"""TabM portfolio scorer (arxiv 2410.24210, ICLR 2025).

Core contribution (from the paper):
  TabM ≈ BatchEnsemble on tabular MLP. A single shared weight matrix W per
  layer is scaled by **per-member rank-1 vectors r ⊙ s^T** (Hadamard) plus a
  per-member bias. k members → k parallel predictions in one forward pass.
  Efficient ensemble: memory ≈ 1 MLP + k·(in+out), predictions averaged.

Concretely for input x ∈ R^D and member i ∈ [0, k):
  h_i = act(  ((x ⊙ r_i^in) W + b) ⊙ s_i^out  + bias_i  )
      = act(W (x ⊙ r_i^in) + bias_i_out) ⊙ s_i^out  (math-equivalent; we use the
        standard BatchEnsemble form below)

Final scalar score per entity = mean of k head predictions.

Per-stock signature: ``model(x: [D], key, inference) -> scalar``.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _act(name: str):
    name = name.lower()
    return {"gelu": jax.nn.gelu, "relu": jax.nn.relu, "silu": jax.nn.silu}[name]


class _BELinear(eqx.Module):
    """BatchEnsemble Linear: shared W, b; per-member (r_in[i], s_out[i], bias[i])."""
    W: jax.Array   # [out, in]
    b: jax.Array   # [out]
    r_in: jax.Array    # [k, in]
    s_out: jax.Array   # [k, out]
    bias_m: jax.Array  # [k, out]

    def __init__(self, in_dim: int, out_dim: int, k: int, *, key: jax.Array):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        # He init for shared W
        self.W = jax.random.normal(k1, (out_dim, in_dim)) * jnp.sqrt(2.0 / in_dim)
        self.b = jnp.zeros((out_dim,))
        # Per-member rank-1 scaling initialized to ones (Wen et al., BatchEnsemble 2020)
        self.r_in = jnp.ones((k, in_dim)) + 0.5 * jax.random.normal(k3, (k, in_dim))
        self.s_out = jnp.ones((k, out_dim)) + 0.5 * jax.random.normal(k4, (k, out_dim))
        self.bias_m = 0.01 * jax.random.normal(k5, (k, out_dim))

    def __call__(self, x_members: jax.Array) -> jax.Array:
        """x_members: [k, in]  → out: [k, out]"""
        # (x_members * r_in) @ W^T + b
        scaled_in = x_members * self.r_in             # [k, in]
        h = scaled_in @ self.W.T + self.b             # [k, out]
        h = h * self.s_out + self.bias_m              # per-member out scaling + bias
        return h


class TabMScorer(eqx.Module):
    layers: list
    dropouts: list
    norms: list
    head: _BELinear
    k: int = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    use_layernorm: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden_units: list[int],
        dropout: float,
        k: int,
        activation: str,
        use_layernorm: bool,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, len(hidden_units) + 1)
        layers = []
        dropouts = []
        norms = []
        prev = in_features
        for i, h in enumerate(hidden_units):
            layers.append(_BELinear(prev, h, k, key=keys[i]))
            dropouts.append(eqx.nn.Dropout(p=dropout))
            norms.append(eqx.nn.LayerNorm(h) if use_layernorm else None)
            prev = h
        self.layers = layers
        self.dropouts = dropouts
        self.norms = norms
        self.head = _BELinear(prev, 1, k, key=keys[-1])
        self.k = k
        self.activation = activation
        self.use_layernorm = use_layernorm

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        """x: [D] → scalar score (mean over k ensemble members)."""
        act = _act(self.activation)
        keys = jax.random.split(key, len(self.layers))
        # Broadcast x to [k, D]
        h = jnp.broadcast_to(x, (self.k, x.shape[-1]))
        for i, (lin, drop, norm) in enumerate(zip(self.layers, self.dropouts, self.norms)):
            h = lin(h)                                  # [k, hidden_i]
            if norm is not None:
                h = jax.vmap(norm)(h)
            h = act(h)
            h = drop(h, key=keys[i], inference=inference)
        out = self.head(h)[:, 0]                        # [k]
        return jnp.mean(out)
