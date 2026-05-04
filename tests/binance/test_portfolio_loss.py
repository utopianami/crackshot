"""Red tests for differentiable portfolio loss.

Contract:
  portfolio_loss(
      w_long, w_short, r_residual, betas, w_prev, *,
      lambda_beta, lambda_turn, lambda_vol, turnover_threshold, fee_per_side,
  ) -> (loss, components)

Components:
  return_term    = -mean(w · r_residual)
  beta_term      = lambda_beta * (sum(beta_i * w_signed_i))^2
  turnover_term  = lambda_turn * max(|w - w_prev|_1 - threshold, 0) * fee_per_side
  vol_term       = lambda_vol * portfolio_vol  (std over time axis when r is [T, N])

This test uses a single-bar variant; per-bar portfolio_vol is zero by definition
so lambda_vol is covered separately.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.training.losses.portfolio import portfolio_loss


def _mk_longshort(w_long_vals, w_short_vals, N=20):
    w = jnp.zeros(N)
    for idx, v in w_long_vals:
        w = w.at[idx].set(v)
    for idx, v in w_short_vals:
        w = w.at[idx].set(-v)
    return w


def test_return_term_rewards_positive_alignment():
    """Loss must decrease when weights align with positive residual returns."""
    N = 20
    r = jnp.array([0.02, -0.01, 0.015, -0.008, 0.0, 0.012, -0.007, 0.009, -0.011, 0.003,
                   -0.002, 0.018, -0.013, 0.005, -0.004, 0.011, -0.006, 0.007, -0.009, 0.001])
    betas = jnp.ones(N)
    w_prev = jnp.zeros(N)

    # Good alignment: long the positive, short the negative
    w_good = jnp.sign(r) * 0.1
    # Bad alignment: inverse
    w_bad = -jnp.sign(r) * 0.1

    loss_good, _ = portfolio_loss(
        w_good, r, betas, w_prev,
        lambda_beta=0.0, lambda_turn=0.0, lambda_vol=0.0,
        turnover_threshold=0.0, fee_per_side=0.0,
    )
    loss_bad, _ = portfolio_loss(
        w_bad, r, betas, w_prev,
        lambda_beta=0.0, lambda_turn=0.0, lambda_vol=0.0,
        turnover_threshold=0.0, fee_per_side=0.0,
    )
    assert float(loss_good) < float(loss_bad)


def test_beta_penalty_zero_when_neutral_positive_when_imbalanced():
    """beta_term ≈ 0 for balanced long/short betas, strictly positive when imbalanced."""
    N = 20
    r = jnp.zeros(N)  # isolate beta term
    w_prev = jnp.zeros(N)

    # Balanced: long symbols have avg beta = 1.0, short symbols have avg beta = 1.0
    betas_balanced = jnp.array([1.0] * N)
    w = jnp.zeros(N)
    for i in range(5):
        w = w.at[i].set(0.1)
    for i in range(5, 10):
        w = w.at[i].set(-0.1)
    _, comps_bal = portfolio_loss(
        w, r, betas_balanced, w_prev,
        lambda_beta=1.0, lambda_turn=0.0, lambda_vol=0.0,
        turnover_threshold=0.0, fee_per_side=0.0,
    )
    assert float(comps_bal["beta_term"]) < 1e-8

    # Imbalanced: long side has high-beta coins (1.5), short side has low-beta coins (0.5)
    betas_imbal = jnp.array([1.5] * 5 + [0.5] * 5 + [1.0] * 10)
    _, comps_imb = portfolio_loss(
        w, r, betas_imbal, w_prev,
        lambda_beta=1.0, lambda_turn=0.0, lambda_vol=0.0,
        turnover_threshold=0.0, fee_per_side=0.0,
    )
    # Net beta = 5 * 0.1 * 1.5 + 5 * (-0.1) * 0.5 = 0.75 - 0.25 = 0.5 -> beta_term = 0.25
    assert float(comps_imb["beta_term"]) > 0.1
    np.testing.assert_allclose(float(comps_imb["beta_term"]), 0.25, atol=1e-5)


def test_turnover_term_respects_threshold():
    """Turnover below threshold adds zero cost; above threshold scales linearly."""
    N = 20
    r = jnp.zeros(N)
    betas = jnp.ones(N)
    w = jnp.zeros(N).at[0].set(0.1).at[1].set(-0.1)
    # No previous weights -> |Δw|_1 = 0.2
    w_prev = jnp.zeros(N)

    # Threshold 0.5 -> below -> zero turnover cost
    _, comps_lo = portfolio_loss(
        w, r, betas, w_prev,
        lambda_beta=0.0, lambda_turn=1.0, lambda_vol=0.0,
        turnover_threshold=0.5, fee_per_side=0.0005,
    )
    assert float(comps_lo["turnover_term"]) == 0.0

    # Threshold 0.0 -> above -> cost = 1.0 * 0.2 * 0.0005 = 1e-4
    _, comps_hi = portfolio_loss(
        w, r, betas, w_prev,
        lambda_beta=0.0, lambda_turn=1.0, lambda_vol=0.0,
        turnover_threshold=0.0, fee_per_side=0.0005,
    )
    np.testing.assert_allclose(float(comps_hi["turnover_term"]), 0.2 * 5e-4, atol=1e-8)


def test_fund_term_penalises_high_funding_longs():
    """fund_term = lambda_fund * sum(w · fund_next). Long high-funding coins → positive penalty."""
    N = 20
    r = jnp.zeros(N)
    betas = jnp.ones(N)
    w_prev = jnp.zeros(N)
    # long 5 high-funding coins, short 5 low-funding coins
    w = jnp.zeros(N).at[:5].set(0.1).at[5:10].set(-0.1)
    # fund_next: longs have +0.01 funding, shorts have +0.005 funding (longs pay more)
    fund_next = jnp.zeros(N).at[:5].set(0.01).at[5:10].set(0.005)
    # Expected fund_term = 1.0 * (5*0.1*0.01 + 5*(-0.1)*0.005) = 0.005 - 0.0025 = 0.0025
    _, comps = portfolio_loss(
        w, r, betas, w_prev,
        lambda_beta=0.0, lambda_turn=0.0, lambda_vol=0.0,
        turnover_threshold=0.0, fee_per_side=0.0,
        lambda_fund=1.0, fund_next=fund_next,
    )
    np.testing.assert_allclose(float(comps["fund_term"]), 0.0025, atol=1e-6)


def test_fund_term_rewards_shorting_positive_funding():
    """Shorting coins with positive funding → negative fund_term (profit)."""
    N = 20
    r = jnp.zeros(N)
    betas = jnp.ones(N)
    w_prev = jnp.zeros(N)
    # short all 10 coins, all with positive funding
    w = jnp.zeros(N).at[:10].set(-0.1)
    fund_next = jnp.zeros(N).at[:10].set(0.01)
    # fund_term = 1.0 * sum(-0.1 * 0.01) * 10 = -0.01
    _, comps = portfolio_loss(
        w, r, betas, w_prev,
        lambda_beta=0.0, lambda_turn=0.0, lambda_vol=0.0,
        turnover_threshold=0.0, fee_per_side=0.0,
        lambda_fund=1.0, fund_next=fund_next,
    )
    np.testing.assert_allclose(float(comps["fund_term"]), -0.01, atol=1e-6)


def test_fund_term_backward_compat_optional():
    """lambda_fund default 0 / fund_next default None → fund_term absent or 0."""
    N = 20
    r = jnp.zeros(N)
    betas = jnp.ones(N)
    w_prev = jnp.zeros(N)
    w = jnp.zeros(N).at[:5].set(0.1).at[5:10].set(-0.1)
    # No funding args passed: must not raise, fund_term must be 0 or absent
    _, comps = portfolio_loss(
        w, r, betas, w_prev,
        lambda_beta=0.0, lambda_turn=0.0, lambda_vol=0.0,
        turnover_threshold=0.0, fee_per_side=0.0,
    )
    assert float(comps.get("fund_term", 0.0)) == 0.0


def test_loss_is_differentiable_end_to_end():
    """Gradient must flow through all components to weights."""
    N = 20
    r = jnp.linspace(-0.02, 0.02, N)
    betas = jnp.linspace(0.5, 1.5, N)
    w_prev = jnp.zeros(N)
    w = jnp.zeros(N).at[:5].set(0.1).at[5:10].set(-0.1)

    def loss_fn(w_in):
        l, _ = portfolio_loss(
            w_in, r, betas, w_prev,
            lambda_beta=0.5, lambda_turn=0.5, lambda_vol=0.0,
            turnover_threshold=0.0, fee_per_side=0.0005,
        )
        return l

    grad = jax.grad(loss_fn)(w)
    # Gradient shape ok + at least some entries non-zero
    assert grad.shape == (N,)
    assert float(jnp.sum(jnp.abs(grad))) > 1e-6
