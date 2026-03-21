"""FT-Transformer (Feature Tokenizer Transformer) with PLE-B tokenization.

Reference: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)

Architecture:
  Input: [batch, F] features
  → PLE-B tokenization: each feature → [d_token] embedding
  → [CLS] token prepended → [batch, F+1, d_token]
  → Transformer Encoder (N layers: pre-norm MHA + FFN)
  → [CLS] token → LayerNorm → Linear → [H, C] logits

Pre-norm Transformer (LayerNorm before each sub-layer).
Inference: softmax → [H, C] probabilities.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.models.architectures.tab_m import (
    PiecewiseLinearEncoding,
    LinearEmbeddings,
    NLinear,
)


# ── Feature Tokenizer (PLE-B per-feature → d_token) ─────────────────────────


class FeatureTokenizer(eqx.Module):
    """Tokenize each feature using PLE-B, then project to d_token.

    PLE-B output per feature: d_embedding dims.
    Projection: Linear(d_embedding → d_token) per feature (shared weight across features).
    """

    ple: PiecewiseLinearEncoding
    linear0: LinearEmbeddings   # residual: scalar → d_embedding
    nlinear: NLinear             # PLE path: n_bins → d_embedding
    proj: jax.Array             # [d_embedding, d_token] shared projection
    proj_bias: jax.Array        # [d_token]
    d_embedding: int
    d_token: int

    def __init__(self, boundaries: np.ndarray, d_embedding: int, d_token: int,
                 *, key: jax.Array):
        num_features = boundaries.shape[0]
        n_bins = boundaries.shape[1] - 1
        self.d_embedding = d_embedding
        self.d_token = d_token

        k_lin, k_proj = jax.random.split(key)
        self.ple = PiecewiseLinearEncoding(boundaries)
        self.linear0 = LinearEmbeddings(num_features, d_embedding, key=k_lin)
        self.nlinear = NLinear(num_features, n_bins, d_embedding)

        # Shared linear projection from d_embedding → d_token
        std = jnp.sqrt(2.0 / (d_embedding + d_token))
        k1, k2 = jax.random.split(k_proj)
        self.proj = jax.random.normal(k1, (d_embedding, d_token)) * std
        self.proj_bias = jnp.zeros((d_token,))

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: [F] → [F, d_token]."""
        ple_out = self.ple(x)                    # [F, n_bins]
        emb = self.linear0(x) + self.nlinear(ple_out)  # [F, d_embedding]
        tokens = emb @ self.proj + self.proj_bias       # [F, d_token]
        return tokens


# ── Transformer components ───────────────────────────────────────────────────


class MultiHeadAttention(eqx.Module):
    """Multi-head scaled dot-product attention (pre-norm applied outside)."""

    q_proj: jax.Array   # [d_token, d_token]
    k_proj: jax.Array
    v_proj: jax.Array
    out_proj: jax.Array
    q_bias: jax.Array   # [d_token]
    k_bias: jax.Array
    v_bias: jax.Array
    out_bias: jax.Array
    n_heads: int
    d_head: int
    attn_dropout: eqx.nn.Dropout

    def __init__(self, d_token: int, n_heads: int, attention_dropout: float, *, key: jax.Array):
        assert d_token % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        d = d_token

        keys = jax.random.split(key, 4)
        std = jnp.sqrt(2.0 / (d + d))
        self.q_proj = jax.random.normal(keys[0], (d, d)) * std
        self.k_proj = jax.random.normal(keys[1], (d, d)) * std
        self.v_proj = jax.random.normal(keys[2], (d, d)) * std
        self.out_proj = jax.random.normal(keys[3], (d, d)) * std
        self.q_bias = jnp.zeros((d,))
        self.k_bias = jnp.zeros((d,))
        self.v_bias = jnp.zeros((d,))
        self.out_bias = jnp.zeros((d,))
        self.attn_dropout = eqx.nn.Dropout(p=attention_dropout)

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool = False) -> jax.Array:
        """x: [S, d_token] → [S, d_token]."""
        S, d = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = x @ self.q_proj + self.q_bias  # [S, d]
        K = x @ self.k_proj + self.k_bias
        V = x @ self.v_proj + self.v_bias

        # Reshape to [H, S, Dh]
        Q = Q.reshape(S, H, Dh).transpose(1, 0, 2)  # [H, S, Dh]
        K = K.reshape(S, H, Dh).transpose(1, 0, 2)
        V = V.reshape(S, H, Dh).transpose(1, 0, 2)

        scale = jnp.sqrt(jnp.array(Dh, dtype=jnp.float32))
        attn = jnp.einsum('hqd,hkd->hqk', Q, K) / scale  # [H, S, S]
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn, key=key, inference=inference)

        out = jnp.einsum('hqs,hsd->qhd', attn, V)  # [S, H, Dh]
        out = out.reshape(S, d)
        out = out @ self.out_proj + self.out_bias
        return out


class TransformerFFN(eqx.Module):
    """Position-wise FFN: Linear → GELU → Dropout → Linear."""

    w1: jax.Array   # [d_token, d_ffn]
    b1: jax.Array   # [d_ffn]
    w2: jax.Array   # [d_ffn, d_token]
    b2: jax.Array   # [d_token]
    dropout: eqx.nn.Dropout

    def __init__(self, d_token: int, d_ffn: int, ffn_dropout: float, *, key: jax.Array):
        k1, k2 = jax.random.split(key)
        std1 = jnp.sqrt(2.0 / (d_token + d_ffn))
        std2 = jnp.sqrt(2.0 / (d_ffn + d_token))
        self.w1 = jax.random.normal(k1, (d_token, d_ffn)) * std1
        self.b1 = jnp.zeros((d_ffn,))
        self.w2 = jax.random.normal(k2, (d_ffn, d_token)) * std2
        self.b2 = jnp.zeros((d_token,))
        self.dropout = eqx.nn.Dropout(p=ffn_dropout)

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool = False) -> jax.Array:
        """x: [S, d_token] → [S, d_token]."""
        h = x @ self.w1 + self.b1
        h = jax.nn.gelu(h)
        h = self.dropout(h, key=key, inference=inference)
        h = h @ self.w2 + self.b2
        return h


class TransformerLayer(eqx.Module):
    """Pre-norm Transformer layer: LN→MHA→residual, LN→FFN→residual."""

    norm1: eqx.nn.LayerNorm
    attn: MultiHeadAttention
    norm2: eqx.nn.LayerNorm
    ffn: TransformerFFN
    residual_dropout: eqx.nn.Dropout

    def __init__(self, d_token: int, n_heads: int, d_ffn: int,
                 attention_dropout: float, ffn_dropout: float, residual_dropout: float,
                 *, key: jax.Array):
        k_attn, k_ffn = jax.random.split(key)
        self.norm1 = eqx.nn.LayerNorm(d_token)
        self.attn = MultiHeadAttention(d_token, n_heads, attention_dropout, key=k_attn)
        self.norm2 = eqx.nn.LayerNorm(d_token)
        self.ffn = TransformerFFN(d_token, d_ffn, ffn_dropout, key=k_ffn)
        self.residual_dropout = eqx.nn.Dropout(p=residual_dropout)

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool = False) -> jax.Array:
        """x: [S, d_token] → [S, d_token]."""
        k1, k2, k3 = jax.random.split(key, 3)

        # MHA sub-layer (pre-norm)
        h = jax.vmap(self.norm1)(x)
        h = self.attn(h, key=k1, inference=inference)
        h = self.residual_dropout(h, key=k2, inference=inference)
        x = x + h

        # FFN sub-layer (pre-norm)
        h = jax.vmap(self.norm2)(x)
        h = self.ffn(h, key=k3, inference=inference)
        x = x + h

        return x


# ── FT-Transformer ───────────────────────────────────────────────────────────


class FTTransformer(eqx.Module):
    """FT-Transformer: PLE-B tokenization + Transformer encoder.

    Each feature is independently tokenized via PLE-B → d_token.
    A learnable [CLS] token is prepended.
    N Transformer layers process the sequence.
    CLS output → LayerNorm → Linear → [H, C] logits.
    """

    tokenizer: FeatureTokenizer
    cls_token: jax.Array          # [d_token]
    layers: list[TransformerLayer]
    output_norm: eqx.nn.LayerNorm
    head: jax.Array               # [d_token, H * C]
    head_bias: jax.Array          # [H * C]

    num_horizons: int
    num_classes: int

    def __init__(
        self,
        num_features: int,
        d_token: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ffn_multiplier: float = 4 / 3,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        d_embedding: int = 16,
        num_classes: int = 5,
        num_horizons: int = 3,
        *,
        ple_boundaries: np.ndarray,
        key: jax.Array,
    ):
        self.num_horizons = num_horizons
        self.num_classes = num_classes

        keys = jax.random.split(key, n_layers + 3)
        k_tok, k_cls, k_head = keys[0], keys[1], keys[2]
        layer_keys = keys[3:]

        self.tokenizer = FeatureTokenizer(
            ple_boundaries, d_embedding, d_token, key=k_tok
        )

        # CLS token: small random init
        self.cls_token = jax.random.normal(k_cls, (d_token,)) * 0.02

        d_ffn = max(1, int(d_token * d_ffn_multiplier))
        self.layers = [
            TransformerLayer(
                d_token, n_heads, d_ffn,
                attention_dropout, ffn_dropout, residual_dropout,
                key=layer_keys[i],
            )
            for i in range(n_layers)
        ]

        self.output_norm = eqx.nn.LayerNorm(d_token)

        # Head
        std = jnp.sqrt(2.0 / (d_token + num_horizons * num_classes))
        self.head = jax.random.normal(k_head, (d_token, num_horizons * num_classes)) * std
        self.head_bias = jnp.zeros((num_horizons * num_classes,))

    def __call__(
        self, x: jax.Array, *, key: jax.Array, inference: bool = False
    ) -> jax.Array:
        """Forward pass for a single sample.

        Training/inference: [H, C] logits (training) or [H, C] probabilities (inference).
        """
        # Tokenize: [F, d_token]
        tokens = self.tokenizer(x)

        # Prepend CLS: [1+F, d_token]
        seq = jnp.concatenate([self.cls_token[None, :], tokens], axis=0)

        # Transformer layers
        layer_keys = jax.random.split(key, len(self.layers))
        for i, layer in enumerate(self.layers):
            seq = layer(seq, key=layer_keys[i], inference=inference)

        # CLS token output
        cls_out = seq[0]  # [d_token]
        cls_out = self.output_norm(cls_out)

        # Head: [H * C] → [H, C]
        logits = cls_out @ self.head + self.head_bias
        logits = logits.reshape(self.num_horizons, self.num_classes)

        if inference:
            return jax.nn.softmax(logits, axis=-1)  # [H, C]
        return logits  # [H, C]
