"""Parametric long/short SOFT top-k head.

Two independent SOFT top-k projections share the score vector:
  - Long leg picks k_long symbols with highest scores   → positive weights.
  - Short leg picks k_short symbols with lowest scores  → negative weights.

Signed output w = w_long - w_short, with w_i in [-1, 1]. k_long and k_short
can differ (asymmetric positioning). Both legs are differentiable via the
underlying log-domain Sinkhorn.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.training.heads.soft_topk import soft_topk


@dataclass
class LongShortResult:
    w: jax.Array
    w_long: jax.Array
    w_short: jax.Array
    eps_used: float
    converged: bool


def long_short_soft_topk(
    scores: jax.Array,
    k_long: int,
    k_short: int,
    eps: float,
    n_iter: int = 100,
    eps_min: float = 0.05,
    eps_max: float = 2.0,
    mask: jax.Array | None = None,
) -> LongShortResult:
    """Parametric long/short SOFT top-k.

    Args:
      scores: [N] real-valued scores per entity.
      k_long / k_short: leg sizes (positive integers).
      mask: optional [N] in {0, 1}. Masked entities are excluded from both
        legs by sending their score to -inf in the long direction and +inf
        (which becomes -inf under -scores) in the short direction, so neither
        leg can put mass on them.
    """
    if mask is not None:
        neg_inf = jnp.full_like(scores, -1e9)
        pos_inf = jnp.full_like(scores, 1e9)
        scores_long = jnp.where(mask > 0, scores, neg_inf)
        scores_short_source = jnp.where(mask > 0, scores, pos_inf)
    else:
        scores_long = scores
        scores_short_source = scores
    long = soft_topk(scores_long, k=k_long, eps=eps, n_iter=n_iter, eps_min=eps_min, eps_max=eps_max)
    short = soft_topk(-scores_short_source, k=k_short, eps=eps, n_iter=n_iter, eps_min=eps_min, eps_max=eps_max)
    w = long.w - short.w
    return LongShortResult(
        w=w,
        w_long=long.w,
        w_short=short.w,
        eps_used=float(max(long.eps_used, short.eps_used)),
        converged=bool(long.converged and short.converged),
    )
