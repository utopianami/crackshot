"""ListNet cross-sectional ranking loss."""

import jax
import jax.numpy as jnp


def jax_rankdata(returns: jax.Array, mask: jax.Array) -> jax.Array:
    """Rank normalize returns within each group, handling padding via mask.

    returns: [N, S] raw returns per timestamp
    mask: [N, S] 1=valid, 0=padding
    Returns: [N, S] ranks normalized to [0, 1]. Padding positions get 0.
    """
    masked_returns = returns + (mask - 1) * 1e9
    ranks = jnp.argsort(jnp.argsort(masked_returns, axis=-1), axis=-1).astype(jnp.float32)
    n_valid = jnp.sum(mask, axis=-1, keepdims=True)
    normalized_ranks = ranks / jnp.maximum(n_valid, 1.0)
    return normalized_ranks * mask


def listnet_loss(
    predicted_scores: jax.Array,
    true_returns: jax.Array,
    mask: jax.Array,
    tau_pred: float = 1.0,
    tau_true: float = 0.1,
) -> jax.Array:
    """Cross-sectional listwise ranking loss (ListNet).

    predicted_scores: [N_ts, S] e.g. P[3]+P[4]-P[0]-P[1] from softmax
    true_returns: [N_ts, S] raw c2c_120 returns
    mask: [N_ts, S] 1=valid, 0=padding
    tau_pred: temperature for predicted score distribution
    tau_true: temperature for true rank distribution (lower = more top/bottom focused)

    Returns: scalar loss, normalized by log(n_valid) to match CE scale (~1-2).
    """
    ranks = jax_rankdata(true_returns, mask)
    neg_inf_mask = (1 - mask) * (-1e9)

    P_true = jax.nn.softmax(ranks / tau_true + neg_inf_mask, axis=-1)
    P_pred = jax.nn.softmax(predicted_scores / tau_pred + neg_inf_mask, axis=-1)

    per_ts_loss = -jnp.sum(P_true * jnp.log(P_pred + 1e-8), axis=-1)
    n_valid = jnp.sum(mask, axis=-1)
    per_ts_loss = per_ts_loss / jnp.maximum(jnp.log(n_valid + 1), 1.0)

    return jnp.mean(per_ts_loss)
