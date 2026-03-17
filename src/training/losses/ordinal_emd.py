"""Earth Mover's Distance loss for ordinal classification."""

import jax
import jax.numpy as jnp


def ordinal_emd_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Earth Mover's Distance loss for ordinal classes.

    Penalizes predictions proportional to distance from true class.
    EMD = sum_k |CDF_pred(k) - CDF_true(k)|

    logits: [N, C] float32
    labels: [N] int32
    """
    n_classes = logits.shape[-1]
    probs = jax.nn.softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, n_classes)
    cdf_pred = jnp.cumsum(probs, axis=-1)
    cdf_true = jnp.cumsum(one_hot, axis=-1)
    emd = jnp.sum(jnp.abs(cdf_pred - cdf_true), axis=-1)
    return jnp.mean(emd)
