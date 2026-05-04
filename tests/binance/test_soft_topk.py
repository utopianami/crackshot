"""Red tests for SOFT differentiable top-k (entropic OT).

SOFT projects scores onto {w : 0 <= w_i <= 1, sum(w) = k}
via log-domain Sinkhorn iteration. Paper: Xie et al. NeurIPS 2020.

Contract:
  soft_topk(scores: [N], k: int, eps: float) -> w: [N]
    w_i in [0, 1] (soft selection)
    sum(w) == k (exact, within tolerance)
    larger score_i => larger w_i (monotone)
    differentiable end-to-end
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.training.heads.soft_topk import soft_topk, SoftTopKResult


def test_soft_topk_marginal_converges_at_eps_min():
    """Sum of soft weights must equal k at eps_min=0.05."""
    scores = jnp.array([3.0, 1.5, -0.2, 2.8, -1.0, 0.7, 2.2, -0.5, 1.1, 0.3,
                        1.9, -0.8, 2.5, 0.0, -1.5, 1.3, 2.1, -0.3, 0.9, -0.1])
    k = 5
    eps = 0.05
    res = soft_topk(scores, k=k, eps=eps, n_iter=200)
    assert isinstance(res, SoftTopKResult)
    w = res.w
    assert w.shape == (20,)
    # Marginal constraint: sum = k
    np.testing.assert_allclose(float(jnp.sum(w)), float(k), atol=1e-3)
    # Each weight in [0, 1] (float32 tolerance)
    assert float(jnp.min(w)) >= -1e-5
    assert float(jnp.max(w)) <= 1.0 + 1e-5
    # Convergence reported
    assert res.converged


def test_soft_topk_gradient_nonzero_on_all_symbols_at_warmup_phase():
    """Large eps spreads gradient to every symbol (no vanishing)."""
    scores = jnp.linspace(-2.0, 2.0, 30)
    k = 10
    eps_warmup = 1.0

    # Use a dot product with a non-constant target so the gradient is informative
    # (jnp.sum(w) itself is fixed to k, so its gradient would be zero by construction)
    target = jnp.sin(jnp.arange(30, dtype=jnp.float32))

    def dot_topk_w(s):
        return jnp.sum(soft_topk(s, k=k, eps=eps_warmup, n_iter=100).w * target)

    grad = jax.grad(dot_topk_w)(scores)
    # Every symbol must have non-trivial gradient at warmup phase
    # (Gumbel top-k would give ~zero to 20/30 symbols)
    assert grad.shape == (30,)
    small_grad_count = int(jnp.sum(jnp.abs(grad) < 1e-8))
    assert small_grad_count <= 2, (
        f"At warmup phase (eps=1.0), gradient must flow to (almost) all symbols; "
        f"got {small_grad_count}/30 with near-zero gradient"
    )


def test_soft_topk_falls_back_on_nonconvergence():
    """When Sinkhorn fails to converge, eps auto-increases and retries."""
    # Pathological input: extreme dynamic range at tight eps
    scores = jnp.array([1e6, -1e6, 0.0, 0.5, -0.3, 1.2, -0.8, 2.1, -1.5, 0.9])
    k = 3
    eps_start = 0.01  # too small for this range
    res = soft_topk(scores, k=k, eps=eps_start, n_iter=10, allow_fallback=True)
    # Either converged at an increased eps, or reports non-convergence (no silent wrong output)
    if res.converged:
        assert res.eps_used > eps_start, "should have bumped eps up to converge"
    # Marginal still respected regardless
    np.testing.assert_allclose(float(jnp.sum(res.w)), float(k), atol=1e-2)


def test_soft_topk_invariance_under_feature_scale_shift():
    """Adding a constant to every score must not change top-k selection.

    BTC/ETH sentinel guard: if this breaks, scale-dominant symbols get pinned.
    """
    scores = jnp.array([1.2, -0.5, 0.8, 2.1, -1.3, 0.4, 1.7, -0.2, 0.9, -0.8,
                        1.5, 0.1, -0.4, 2.3, -0.9, 0.6, 1.0, -0.1, 0.3, -1.1])
    k = 5
    eps = 0.1
    w_orig = soft_topk(scores, k=k, eps=eps, n_iter=200).w
    w_shifted = soft_topk(scores + 5.0, k=k, eps=eps, n_iter=200).w
    # Invariant: uniform shift cannot change relative ranking → w must match
    np.testing.assert_allclose(np.asarray(w_orig), np.asarray(w_shifted), atol=1e-3)
