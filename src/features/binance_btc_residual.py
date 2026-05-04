"""BTC-residual return features.

For each coin's 1-bar return series, compute a rolling OLS beta against
BTC's 1-bar return, then produce horizon-H residual returns as the
cumulative residual over the past H bars.

Residual_H(t) = sum_{k=t-H+1..t} ( r_c(k) - β(k) × r_btc(k) )

β(k) is the rolling OLS slope of coin on BTC over the past `beta_window`
1-bar observations (default 2016 = 7d of 5m bars).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_beta(
    coin: np.ndarray,
    btc: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Rolling OLS beta of `coin` on `btc` over `window` observations.

    Uses pandas rolling cov / var. Returns NaN for the first (window-1)
    bars and anywhere btc rolling variance is zero.
    """
    s_c = pd.Series(coin, dtype=float)
    s_b = pd.Series(btc, dtype=float)
    cov = s_c.rolling(window, min_periods=window).cov(s_b)
    var = s_b.rolling(window, min_periods=window).var()
    var = var.where(var > 0, np.nan)
    beta = cov / var
    return beta.to_numpy()


def compute_residual_returns(
    *,
    coin_1bar: np.ndarray,
    btc_1bar: np.ndarray,
    horizons: dict[str, int] | None = None,
    beta_window: int = 2016,
) -> pd.DataFrame:
    """Compute horizon-H BTC-residual returns for a coin.

    Inputs must be aligned (same length, same timeline). Returns a
    DataFrame with columns ``res_ret_{label}`` for each horizon plus
    an internal ``res_ret_1bar`` column.
    """
    if horizons is None:
        horizons = {"1h": 12, "4h": 48, "24h": 288}

    if len(coin_1bar) != len(btc_1bar):
        raise ValueError(
            f"Length mismatch: coin={len(coin_1bar)} btc={len(btc_1bar)}"
        )

    coin = np.asarray(coin_1bar, dtype=float)
    btc = np.asarray(btc_1bar, dtype=float)

    beta = rolling_beta(coin, btc, window=beta_window)
    res_1bar = coin - beta * btc

    out = pd.DataFrame({"res_ret_1bar": res_1bar})
    for label, h in horizons.items():
        cum = (
            pd.Series(res_1bar)
            .rolling(h, min_periods=h)
            .sum()
            .to_numpy()
        )
        out[f"res_ret_{label}"] = cum
    return out


def compute_forward_residual_returns(
    *,
    coin_1bar: np.ndarray,
    btc_1bar: np.ndarray,
    horizons: dict[str, int] | None = None,
    beta_window: int = 2016,
) -> pd.DataFrame:
    """Forward β-residual returns (training target) + β exposure at time t.

    At each timestamp t, freeze β = β(t) from the rolling OLS ending at t
    (uses only past data — no look-ahead). Then the forward residual return
    over horizon H is:

        fwd_res_H(t) = Σ_{k=t+1..t+H} ( r_coin(k) - β(t) · r_btc(k) )

    This is the natural training target for a long/short portfolio that
    hedges against BTC at its current β — matches how the portfolio loss
    scores weights at t.

    Returns a DataFrame with columns ``fwd_res_{label}`` per horizon
    and a ``beta`` column. Last H rows of each fwd_res_H are NaN
    (no future data).
    """
    if horizons is None:
        horizons = {"1h": 12, "4h": 48, "24h": 288}

    if len(coin_1bar) != len(btc_1bar):
        raise ValueError(
            f"Length mismatch: coin={len(coin_1bar)} btc={len(btc_1bar)}"
        )

    coin = np.asarray(coin_1bar, dtype=float)
    btc = np.asarray(btc_1bar, dtype=float)
    T = len(coin)

    beta = rolling_beta(coin, btc, window=beta_window)

    out = pd.DataFrame({"beta": beta})
    for label, h in horizons.items():
        fwd = np.full(T, np.nan, dtype=float)
        # Vectorized forward-sum using cumsum on coin and btc.
        # sum_{k=t+1..t+h} coin[k] = cumsum_coin[t+h] - cumsum_coin[t]
        cs_c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(coin))])
        cs_b = np.concatenate([[0.0], np.cumsum(np.nan_to_num(btc))])
        # But need to preserve NaNs propagation for future-window if any raw NaN
        # Use mask: if any NaN in the future window → NaN.
        valid_coin = ~np.isnan(coin)
        valid_btc = ~np.isnan(btc)
        cs_valid_c = np.concatenate([[0], np.cumsum(valid_coin)])
        cs_valid_b = np.concatenate([[0], np.cumsum(valid_btc)])

        # Valid indices where t+h exists and β(t) is finite
        t_idx = np.arange(T - h)
        sum_c = cs_c[t_idx + h + 1] - cs_c[t_idx + 1]  # k=t+1..t+h inclusive
        sum_b = cs_b[t_idx + h + 1] - cs_b[t_idx + 1]
        all_c_valid = (cs_valid_c[t_idx + h + 1] - cs_valid_c[t_idx + 1]) == h
        all_b_valid = (cs_valid_b[t_idx + h + 1] - cs_valid_b[t_idx + 1]) == h
        vals = sum_c - beta[t_idx] * sum_b
        ok = all_c_valid & all_b_valid & np.isfinite(beta[t_idx])
        fwd[t_idx] = np.where(ok, vals, np.nan)
        out[f"fwd_res_{label}"] = fwd

    return out
