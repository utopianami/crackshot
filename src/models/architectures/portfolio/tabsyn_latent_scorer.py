"""Tabsyn-latent portfolio scorer — VAE-regularized feature encoder.

Paper: Zhang et al. 2023, "Mixed-Type Tabular Data Synthesis with Score-based
Diffusion in Latent Space" (arxiv 2310.09656, ICLR 2024).

Contribution (numeric-only adaptation for portfolio scoring):
  A Transformer VAE encodes each stock's feature vector into a latent that
  captures inter-column relationships. The encoder's pooled latent replaces
  the raw input to the scorer head. We drop the diffusion synthesis stage of
  the original paper — we only keep the VAE-regularized encoder.

Architecture:
  Per-feature Tokenizer: t_i = W_i · x_i + b_i           [D, d_token]
  Prepend learnable [CLS]                                [D+1, d_token]
  Encoder Transformer (n_enc_layers)                     [D+1, d_token]
  Per-token μ-head / logσ²-head (Linear d_token→d_latent)
  Reparameterize z = μ + σε (train) | z = μ (infer)      [D+1, d_latent]
  Decoder input: Linear(d_latent→d_token)                [D+1, d_token]
  Decoder Transformer (n_dec_layers)                     [D+1, d_token]
  Reconstructor per-token Linear(d_token→1)              [D+1]→drop CLS→x̂[D]
  Scorer head: mean-pool z over tokens → MLP             scalar

VAE aux loss:
  total = MSE(x̂, x) + kl_beta · KL(N(μ,σ²) || N(0,I))
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class _FeatureTokenizer(eqx.Module):
    W: jax.Array
    b: jax.Array

    def __init__(self, n_features: int, d_token: int, *, key: jax.Array):
        k1, _ = jax.random.split(key, 2)
        self.W = jax.random.normal(k1, (n_features, d_token)) * (1.0 / jnp.sqrt(d_token))
        self.b = jnp.zeros((n_features, d_token))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x[:, None] * self.W + self.b


class _TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    ffn1: eqx.nn.Linear
    ffn2: eqx.nn.Linear
    drop: eqx.nn.Dropout

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float, *, key: jax.Array):
        k1, k2, k3, _ = jax.random.split(key, 4)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.attn = eqx.nn.MultiheadAttention(num_heads=n_heads, query_size=d_model, key=k1)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.ffn1 = eqx.nn.Linear(d_model, d_model * ffn_mult, key=k2)
        self.ffn2 = eqx.nn.Linear(d_model * ffn_mult, d_model, key=k3)
        self.drop = eqx.nn.Dropout(dropout)

    def __call__(self, tokens: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
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


class _MLPHead(eqx.Module):
    layers: list

    def __init__(self, in_dim: int, hidden_units: list[int], *, key: jax.Array):
        dims = [in_dim] + list(hidden_units) + [1]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
                       for i in range(len(dims) - 1)]

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1:
                x = jax.nn.gelu(x)
        return x


class TabsynLatentScorer(eqx.Module):
    tokenizer: _FeatureTokenizer
    cls_token: jax.Array
    enc_blocks: list
    mu_head: eqx.nn.Linear
    logvar_head: eqx.nn.Linear
    dec_in: eqx.nn.Linear
    dec_blocks: list
    reconstructor: eqx.nn.Linear
    head_norm: eqx.nn.LayerNorm
    head: _MLPHead
    kl_beta: float = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    d_latent: int = eqx.field(static=True)

    def __init__(
        self,
        n_features: int,
        d_token: int,
        d_latent: int,
        n_enc_layers: int,
        n_dec_layers: int,
        n_heads: int,
        ffn_mult: int,
        dropout: float,
        hidden_units: list[int],
        kl_beta: float,
        *,
        key: jax.Array,
    ):
        n_total = 6 + n_enc_layers + n_dec_layers + 1  # +1 for head
        keys = jax.random.split(key, n_total)
        k_tok, k_cls, k_mu, k_lv, k_din, k_rec = keys[:6]
        k_enc = keys[6:6 + n_enc_layers]
        k_dec = keys[6 + n_enc_layers:6 + n_enc_layers + n_dec_layers]
        k_head = keys[-1]

        self.tokenizer = _FeatureTokenizer(n_features, d_token, key=k_tok)
        self.cls_token = 0.02 * jax.random.normal(k_cls, (d_token,))
        self.enc_blocks = [
            _TransformerBlock(d_token, n_heads, ffn_mult, dropout, key=k_enc[i])
            for i in range(n_enc_layers)
        ]
        self.mu_head = eqx.nn.Linear(d_token, d_latent, key=k_mu)
        self.logvar_head = eqx.nn.Linear(d_token, d_latent, key=k_lv)
        self.dec_in = eqx.nn.Linear(d_latent, d_token, key=k_din)
        self.dec_blocks = [
            _TransformerBlock(d_token, n_heads, ffn_mult, dropout, key=k_dec[i])
            for i in range(n_dec_layers)
        ]
        self.reconstructor = eqx.nn.Linear(d_token, 1, key=k_rec)
        self.head_norm = eqx.nn.LayerNorm(d_latent)
        self.head = _MLPHead(d_latent, list(hidden_units), key=k_head)
        self.kl_beta = float(kl_beta)
        self.n_features = n_features
        self.d_latent = d_latent

    def _encode(self, x: jax.Array, *, key: jax.Array, inference: bool):
        tokens = self.tokenizer(x)
        tokens = jnp.concatenate([self.cls_token[None, :], tokens], axis=0)
        keys = jax.random.split(key, max(1, len(self.enc_blocks)))
        for i, blk in enumerate(self.enc_blocks):
            tokens = blk(tokens, key=keys[i], inference=inference)
        mu = jax.vmap(self.mu_head)(tokens)
        logvar = jax.vmap(self.logvar_head)(tokens)
        return mu, logvar

    def _reparameterize(self, mu: jax.Array, logvar: jax.Array, *, key: jax.Array, inference: bool):
        if inference:
            return mu
        sigma = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        return mu + sigma * eps

    def _decode(self, z: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        tokens = jax.vmap(self.dec_in)(z)
        keys = jax.random.split(key, max(1, len(self.dec_blocks)))
        for i, blk in enumerate(self.dec_blocks):
            tokens = blk(tokens, key=keys[i], inference=inference)
        x_hat = jax.vmap(self.reconstructor)(tokens).squeeze(-1)  # [D+1]
        return x_hat[1:]  # drop CLS

    def __call__(self, x: jax.Array, *, key: jax.Array, inference: bool) -> jax.Array:
        k_enc, k_rep = jax.random.split(key, 2)
        mu, logvar = self._encode(x, key=k_enc, inference=inference)
        z = self._reparameterize(mu, logvar, key=k_rep, inference=inference)
        pooled = jnp.mean(z[1:], axis=0)  # mean over non-CLS tokens
        pooled = self.head_norm(pooled)
        return self.head(pooled)[0]

    def aux_loss(self, x: jax.Array, *, key: jax.Array) -> dict:
        k_enc, k_rep, k_dec = jax.random.split(key, 3)
        mu, logvar = self._encode(x, key=k_enc, inference=False)
        z = self._reparameterize(mu, logvar, key=k_rep, inference=False)
        x_hat = self._decode(z, key=k_dec, inference=False)
        mse = jnp.mean((x_hat - x) ** 2)
        kl = -0.5 * jnp.mean(1.0 + logvar - mu ** 2 - jnp.exp(logvar))
        total = mse + self.kl_beta * kl
        return {"mse": mse, "kl": kl, "total": total}
