"""Red tests for forward residual returns + β exposure.

Training target for the differentiable portfolio loss is the β-residual
forward return over horizon H:

  fwd_res_H(t) = sum_{k=t+1..t+H} ( r_coin(k) - β(t) * r_btc(k) )

β(t) = rolling OLS slope of coin on BTC over the most recent `beta_window`
1-bar returns ending at t (no look-ahead).

Also exposes β(t) as a column so the portfolio loss can enforce β-neutrality.
"""
from __future__ import annotations

import numpy as np

from src.features.binance_btc_residual import (
    compute_forward_residual_returns,
)


def test_forward_residual_no_lookahead_shape():
    """Result must have one row per input timestamp; last H rows are NaN."""
    np.random.seed(0)
    T = 4000
    coin = np.random.randn(T) * 0.01
    btc = 0.7 * coin + np.random.randn(T) * 0.005

    out = compute_forward_residual_returns(
        coin_1bar=coin, btc_1bar=btc,
        horizons={"1h": 12, "4h": 48, "24h": 288},
        beta_window=2016,
    )
    assert "fwd_res_1h" in out.columns
    assert "fwd_res_4h" in out.columns
    assert "fwd_res_24h" in out.columns
    assert "beta" in out.columns
    # Last H rows for each horizon must be NaN (no future available)
    assert np.isnan(out["fwd_res_1h"].iloc[-1])
    assert np.isnan(out["fwd_res_4h"].iloc[-1])
    # Length preserved
    assert len(out) == T


def test_forward_residual_zero_beta_equals_coin_return():
    """When β ≈ 0, fwd residual should equal cumulative coin return (forward sum)."""
    np.random.seed(1)
    T = 3000
    coin = np.random.randn(T) * 0.01
    # Orthogonal BTC → β ≈ 0
    btc = np.random.randn(T) * 0.01

    out = compute_forward_residual_returns(
        coin_1bar=coin, btc_1bar=btc,
        horizons={"1h": 12},
        beta_window=2016,
    )
    # Pick a middle timestamp with valid β
    t = 2500
    beta_t = out["beta"].iloc[t]
    fwd_res = out["fwd_res_1h"].iloc[t]
    # Expected = sum of (coin - beta * btc) over t+1..t+12
    expected = float(np.sum(coin[t + 1: t + 13] - beta_t * btc[t + 1: t + 13]))
    np.testing.assert_allclose(float(fwd_res), expected, atol=1e-7)


def test_beta_series_matches_rolling_ols():
    """β series must equal rolling_beta() output on the same window."""
    from src.features.binance_btc_residual import rolling_beta
    np.random.seed(2)
    T = 5000
    coin = np.random.randn(T) * 0.01
    btc = 1.2 * coin + np.random.randn(T) * 0.005

    out = compute_forward_residual_returns(
        coin_1bar=coin, btc_1bar=btc,
        horizons={"1h": 12},
        beta_window=2016,
    )
    ref = rolling_beta(coin, btc, window=2016)
    np.testing.assert_allclose(out["beta"].to_numpy(), ref, atol=1e-10, equal_nan=True)
