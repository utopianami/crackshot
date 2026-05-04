"""Differentiable portfolio loss for cross-sectional long/short.

loss = -mean(w · r_residual)                       # β-residual portfolio return
     + lambda_beta * (sum(beta_i · w_i))^2         # β-neutrality soft constraint
     + lambda_turn * max(|Δw|_1 - threshold, 0) * fee_per_side
     + lambda_vol  * portfolio_vol                 # optional, 0 for single-bar

All terms are differentiable. Volatility term is returned as a component but
the caller is expected to aggregate it across bars when computing the time-series
variance; per-bar it is zero. Funding PnL is handled in evaluation, not training.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def portfolio_loss(
    w: jax.Array,
    r_residual: jax.Array,
    betas: jax.Array,
    w_prev: jax.Array,
    *,
    lambda_beta: float,
    lambda_turn: float,
    lambda_vol: float,
    turnover_threshold: float,
    fee_per_side: float,
    lambda_fund: float = 0.0,
    fund_next: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Per-bar portfolio loss.

    Args:
      w: [N] signed weights (long: positive, short: negative).
      r_residual: [N] β-residual returns for the next bar.
      betas: [N] per-symbol BTC beta.
      w_prev: [N] signed weights at previous bar (for turnover).
      lambda_beta, lambda_turn, lambda_vol: penalty strengths.
      turnover_threshold: free budget before turnover cost kicks in (fraction).
      fee_per_side: round-trip transaction cost per unit of weight (e.g. 5e-4 for 5bp).
      lambda_fund: funding-cost penalty weight. Default 0 (term disabled).
      fund_next: [N] per-symbol sum of fundingRate over the next bar horizon.
                 Positive → longs pay, shorts receive.
    """
    port_ret = jnp.sum(w * r_residual)
    return_term = -port_ret

    net_beta = jnp.sum(betas * w)
    beta_term = lambda_beta * net_beta * net_beta

    dw_l1 = jnp.sum(jnp.abs(w - w_prev))
    excess = jnp.maximum(dw_l1 - turnover_threshold, 0.0)
    turnover_term = lambda_turn * excess * fee_per_side

    vol_term = lambda_vol * 0.0

    if fund_next is None:
        fund_term = jnp.asarray(0.0)
    else:
        fund_term = lambda_fund * jnp.sum(w * fund_next)

    total = return_term + beta_term + turnover_term + vol_term + fund_term
    components = {
        "return_term": return_term,
        "beta_term": beta_term,
        "turnover_term": turnover_term,
        "vol_term": vol_term,
        "fund_term": fund_term,
        "port_ret": port_ret,
        "net_beta": net_beta,
        "turnover_l1": dw_l1,
    }
    return total, components
