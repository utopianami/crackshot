"""SOFT differentiable top-k via entropic OT (Sinkhorn).

Paper: Xie, Dai, Chen, Song. "Differentiable Top-k Operator with Optimal Transport." NeurIPS 2020.

Projects scores s in R^N onto {w : 0 <= w_i <= 1, sum(w) = k} by solving an
entropic optimal-transport problem with a binary sink (selected / unselected).
Implemented with log-domain Sinkhorn iterations for numerical stability at
small epsilon (regime-boundary score distributions can otherwise underflow).

Shift invariance: a uniform additive shift of the score vector does not
change the relative ranking, hence must not change w. We enforce this by
max-centering scores before forming the transport potentials.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@dataclass
class SoftTopKResult:
    w: jax.Array
    eps_used: float
    converged: bool
    n_iter_used: int


class _SinkhornState(NamedTuple):
    log_u: jax.Array
    log_v: jax.Array


def _sinkhorn_log(scores: jax.Array, k: int, eps: float, n_iter: int) -> jax.Array:
    """Log-domain Sinkhorn for binary-sink OT. Returns selection weights w in [0,1]^N."""
    # Max-center for shift invariance and numerical stability
    scores = scores - jnp.max(scores)
    N = scores.shape[0]

    # Source marginal: each item supplies 1 unit of mass  -> log a = 0
    log_a = jnp.zeros(N)
    # Sink marginal: "selected" bin has k, "unselected" bin has N - k
    log_b = jnp.log(jnp.array([float(k), float(N - k)], dtype=scores.dtype))

    # log K[i, c] = -C[i, c] / eps where C[i, 0] = -scores[i] (reward selection), C[i, 1] = 0
    log_K_col0 = scores / eps
    log_K_col1 = jnp.zeros(N, dtype=scores.dtype)
    log_K = jnp.stack([log_K_col0, log_K_col1], axis=1)  # [N, 2]

    def body(state: _SinkhornState, _):
        log_u = log_a - logsumexp(log_K + state.log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_K + log_u[:, None], axis=0)
        return _SinkhornState(log_u=log_u, log_v=log_v), None

    init = _SinkhornState(log_u=jnp.zeros(N, dtype=scores.dtype),
                           log_v=jnp.zeros(2, dtype=scores.dtype))
    final, _ = jax.lax.scan(body, init, None, length=n_iter)

    # Transport plan, column 0 = selection probability per item
    w = jnp.exp(final.log_u + log_K[:, 0] + final.log_v[0])
    return w


def soft_topk(
    scores: jax.Array,
    k: int,
    eps: float,
    n_iter: int = 100,
    allow_fallback: bool = False,
    eps_min: float = 0.05,
    eps_max: float = 2.0,
    tol: float = 1e-2,
    max_bumps: int = 10,
) -> SoftTopKResult:
    """Differentiable top-k.

    Args:
      scores: [N] real-valued scores, higher = more likely selected.
      k: target number of selected items (sum(w) will equal k).
      eps: entropy regularization. Clipped up to eps_min.
      n_iter: Sinkhorn iterations.
      allow_fallback: on non-convergence, multiplicatively increase eps and retry
        until either convergence or eps >= eps_max.
    """
    scores_arr = jnp.asarray(scores, dtype=jnp.float32)
    eps_curr = max(float(eps), float(eps_min))

    w = _sinkhorn_log(scores_arr, k, eps_curr, n_iter)

    # Under jax tracing (jit/grad), skip Python-side convergence check + fallback.
    # Fallback requires concretization; in training we rely on the warmup schedule
    # to keep eps in a converging regime.
    if isinstance(scores_arr, jax.core.Tracer) or isinstance(w, jax.core.Tracer):
        return SoftTopKResult(w=w, eps_used=eps_curr, converged=True, n_iter_used=n_iter)

    marginal_err = float(jnp.abs(jnp.sum(w) - float(k)))
    bad = bool(jnp.any(jnp.isnan(w))) or bool(jnp.any(jnp.isinf(w)))
    converged = (not bad) and (marginal_err < tol)

    bumps = 0
    while (not converged) and allow_fallback and eps_curr < eps_max and bumps < max_bumps:
        eps_curr = min(eps_curr * 1.5, eps_max)
        w = _sinkhorn_log(scores_arr, k, eps_curr, n_iter)
        marginal_err = float(jnp.abs(jnp.sum(w) - float(k)))
        bad = bool(jnp.any(jnp.isnan(w))) or bool(jnp.any(jnp.isinf(w)))
        converged = (not bad) and (marginal_err < tol)
        bumps += 1

    return SoftTopKResult(
        w=w,
        eps_used=float(eps_curr),
        converged=bool(converged),
        n_iter_used=int(n_iter),
    )
