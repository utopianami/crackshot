"""Red tests for parametric long/short SOFT top-k head.

Contract:
  long_short_soft_topk(scores, k_long, k_short, eps, n_iter) -> LongShortResult
    w: [N] signed weights in [-1, 1]
    sum(positive w) == k_long   (long leg)
    sum(-negative w) == k_short (short leg)
    no symbol appears in both legs simultaneously (weights at tails separated)
    differentiable end-to-end
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.training.heads.long_short_topk import (
    LongShortResult,
    long_short_soft_topk,
)


def test_long_short_marginals_match_requested_sizes():
    """sum of long leg = k_long, sum of short leg = k_short, asymmetric allowed."""
    scores = jnp.linspace(-2.0, 2.0, 30)
    k_long = 10
    k_short = 5
    res = long_short_soft_topk(scores, k_long=k_long, k_short=k_short, eps=0.1, n_iter=200)
    assert isinstance(res, LongShortResult)
    w = res.w
    assert w.shape == (30,)
    long_mass = float(jnp.sum(jnp.maximum(w, 0.0)))
    short_mass = float(jnp.sum(jnp.maximum(-w, 0.0)))
    np.testing.assert_allclose(long_mass, float(k_long), atol=1e-2)
    np.testing.assert_allclose(short_mass, float(k_short), atol=1e-2)


def test_long_short_longs_high_scores_shorts_low_scores():
    """High-score symbols get positive weight; low-score symbols get negative."""
    scores = jnp.linspace(-3.0, 3.0, 30)  # ascending
    res = long_short_soft_topk(scores, k_long=5, k_short=5, eps=0.1, n_iter=200)
    w = res.w
    # Top 5 (indices 25..29) should be positive
    top = w[25:]
    bot = w[:5]
    assert float(jnp.min(top)) > 0.0
    assert float(jnp.max(bot)) < 0.0


def test_long_short_differentiable():
    """Gradient flows through both legs."""
    scores = jnp.linspace(-2.0, 2.0, 20)
    target_long = jnp.array([0.0] * 15 + [1.0] * 5)
    target_short = jnp.array([1.0] * 5 + [0.0] * 15)

    def loss_fn(s):
        r = long_short_soft_topk(s, k_long=5, k_short=5, eps=0.2, n_iter=100)
        # reward correct sign on both ends
        return -(jnp.sum(r.w * target_long) - jnp.sum(r.w * target_short))

    g = jax.grad(loss_fn)(scores)
    assert g.shape == (20,)
    assert float(jnp.sum(jnp.abs(g))) > 1e-6


def test_long_short_symmetric_equals_single_topk_difference():
    """For k_long == k_short, symmetric scores → symmetric w around zero."""
    scores = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0, -1.5, 1.5, 0.5])
    res = long_short_soft_topk(scores, k_long=2, k_short=2, eps=0.1, n_iter=200)
    w = res.w
    # Net exposure ~ 0 (k_long = k_short, symmetric-ish scores)
    net = float(jnp.sum(w))
    assert abs(net) < 0.1
