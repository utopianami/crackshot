"""Label smoothing cross-entropy with optional focal weighting."""

import jax
import jax.numpy as jnp


def label_smoothing_ce(
    logits: jax.Array, labels: jax.Array, smoothing: float = 0.05,
    class_weights: jax.Array | None = None,
    focal_gamma: float = 0.0,
) -> jax.Array:
    """Cross-entropy with label smoothing, optional focal weighting, and per-class weighting.

    logits: [N, C] float32
    labels: [N] int32
    class_weights: [C] float32, optional. Higher weight = more emphasis on that class.
    focal_gamma: float. 0=standard CE. >0 applies focal weight (1-p_t)^gamma.
    """
    n_classes = logits.shape[-1]
    smooth_val = smoothing / n_classes
    one_hot = jax.nn.one_hot(labels, n_classes)
    soft_labels = one_hot * (1.0 - smoothing) + smooth_val
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    if focal_gamma > 0:
        probs = jax.nn.softmax(logits, axis=-1)
        focal_weight = (1.0 - probs) ** focal_gamma
        per_sample = -jnp.sum(focal_weight * soft_labels * log_probs, axis=-1)
    else:
        per_sample = -jnp.sum(soft_labels * log_probs, axis=-1)
    if class_weights is not None:
        sample_weights = jnp.take(class_weights, labels)
        return jnp.mean(per_sample * sample_weights)
    return jnp.mean(per_sample)
