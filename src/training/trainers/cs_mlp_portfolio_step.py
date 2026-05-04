"""JIT-compiled training step for the cs_mlp_portfolio trainer.

Exposed as a pure factory so unit tests can call it without the whole
trainer scaffolding.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from src.training.heads.long_short_topk import long_short_soft_topk
from src.training.losses.portfolio import portfolio_loss


def _scores_per_ts(model, x_ts, key, inference: bool):
    """Apply the scorer to every padded entity → [N_ts, S].

    Dispatches on the scorer's ``cross_entity_scoring`` attribute:
      - False (default): per-stock scorer, ``model(x_i: [D]) → scalar``, vmap'd over S internally.
      - True: cross-entity scorer (e.g. MASTER attention), ``model(x_stocks: [S, D]) → [S]``.
    """
    is_cross_entity = bool(getattr(model, "cross_entity_scoring", False))
    if is_cross_entity:
        keys_ts = jax.random.split(key, x_ts.shape[0])

        def per_ts_ce(x_stocks, k):
            return model(x_stocks, key=k, inference=inference)

        return jax.vmap(per_ts_ce)(x_ts, keys_ts)

    def per_stock(xi, ki):
        return model(xi, key=ki, inference=inference)

    def per_ts(x_stocks, keys_stocks):
        return jax.vmap(per_stock)(x_stocks, keys_stocks)

    keys = jax.random.split(key, x_ts.shape[0] * x_ts.shape[1])
    keys = keys.reshape(x_ts.shape[0], x_ts.shape[1], 2)
    return jax.vmap(per_ts)(x_ts, keys)


def _weights_per_ts(scores, mask_ts, k_long, k_short, eps, sinkhorn_iter):
    """Project per-ts scores → signed weights via long/short SOFT top-k (mask-aware)."""
    def per_ts(s_row, m_row):
        out = long_short_soft_topk(
            s_row, k_long=k_long, k_short=k_short, eps=eps,
            n_iter=sinkhorn_iter, mask=m_row,
        )
        return out.w

    w = jax.vmap(per_ts)(scores, mask_ts)
    return w * mask_ts  # belt-and-suspenders — mask inside the head already prevents pad selection


def _loss_per_ts(w_ts, r_ts, betas_ts, w_prev_ts, fund_ts=None, **lp_kwargs):
    """Apply portfolio_loss at every timestamp and average."""
    if fund_ts is None:
        def per_ts(w, r, betas, w_prev):
            return portfolio_loss(w, r, betas, w_prev, **lp_kwargs)
        losses, comps = jax.vmap(per_ts)(w_ts, r_ts, betas_ts, w_prev_ts)
    else:
        def per_ts(w, r, betas, w_prev, fund):
            return portfolio_loss(w, r, betas, w_prev, fund_next=fund, **lp_kwargs)
        losses, comps = jax.vmap(per_ts)(w_ts, r_ts, betas_ts, w_prev_ts, fund_ts)
    mean_loss = jnp.mean(losses)
    aux = {
        "mean_port_ret": jnp.mean(comps["port_ret"]),
        "mean_turnover_l1": jnp.mean(comps["turnover_l1"]),
        "mean_net_beta": jnp.mean(comps["net_beta"]),
        "mean_return_term": jnp.mean(comps["return_term"]),
        "mean_beta_term": jnp.mean(comps["beta_term"]),
        "mean_turnover_term": jnp.mean(comps["turnover_term"]),
        "mean_fund_term": jnp.mean(comps["fund_term"]),
    }
    return mean_loss, aux


def _scorer_aux_loss(model, x_ts, mask_ts, key):
    """Mean per-stock aux_loss over padded batch, mask-weighted.

    Used by scorers that expose ``aux_loss(x, *, key) -> {'total', ...}``
    (e.g. Tabsyn VAE reconstruction). Returns 0 if the scorer does not
    implement aux_loss.
    """
    if not hasattr(model, "aux_loss"):
        return jnp.asarray(0.0)
    N_ts, S = x_ts.shape[0], x_ts.shape[1]
    keys = jax.random.split(key, N_ts * S).reshape(N_ts, S, 2)

    def per_stock(xi, mi, ki):
        return model.aux_loss(xi, key=ki)["total"] * mi

    def per_ts(xs, ms, ks):
        return jax.vmap(per_stock)(xs, ms, ks)

    tot = jax.vmap(per_ts)(x_ts, mask_ts, keys)
    denom = jnp.maximum(jnp.sum(mask_ts), 1.0)
    return jnp.sum(tot) / denom


def portfolio_train_step_fn(
    optimizer,
    *,
    k_long: int,
    k_short: int,
    eps: float,
    sinkhorn_iter: int,
    lambda_beta: float,
    lambda_turn: float,
    lambda_vol: float,
    turnover_threshold: float,
    fee_per_side: float,
    lambda_fund: float = 0.0,
    lambda_aux: float = 0.0,
):
    lp_kwargs = dict(
        lambda_beta=lambda_beta,
        lambda_turn=lambda_turn,
        lambda_vol=lambda_vol,
        turnover_threshold=turnover_threshold,
        fee_per_side=fee_per_side,
        lambda_fund=lambda_fund,
    )

    @eqx.filter_jit
    def step(model, opt_state, x_ts, betas_ts, r_res_ts, w_prev_ts, mask_ts, key,
             fund_next_ts=None):
        def loss_fn(model):
            k_sc, k_aux = jax.random.split(key, 2)
            scores = _scores_per_ts(model, x_ts, k_sc, inference=False)
            w = _weights_per_ts(scores, mask_ts, k_long, k_short, eps, sinkhorn_iter)
            port_loss, aux = _loss_per_ts(w, r_res_ts, betas_ts, w_prev_ts,
                                          fund_ts=fund_next_ts, **lp_kwargs)
            if lambda_aux > 0.0:
                vae_aux = _scorer_aux_loss(model, x_ts, mask_ts, k_aux)
                port_loss = port_loss + lambda_aux * vae_aux
                aux = {**aux, "vae_aux_total": vae_aux}
            return port_loss, aux

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt, loss, aux

    return step


def portfolio_eval_step_fn(
    *,
    k_long: int,
    k_short: int,
    eps: float,
    sinkhorn_iter: int,
    lambda_beta: float,
    lambda_turn: float,
    lambda_vol: float,
    turnover_threshold: float,
    fee_per_side: float,
    lambda_fund: float = 0.0,
):
    lp_kwargs = dict(
        lambda_beta=lambda_beta,
        lambda_turn=lambda_turn,
        lambda_vol=lambda_vol,
        turnover_threshold=turnover_threshold,
        fee_per_side=fee_per_side,
        lambda_fund=lambda_fund,
    )

    @eqx.filter_jit
    def step(model, x_ts, betas_ts, r_res_ts, w_prev_ts, mask_ts, key,
             fund_next_ts=None):
        scores = _scores_per_ts(model, x_ts, key, inference=True)
        w = _weights_per_ts(scores, mask_ts, k_long, k_short, eps, sinkhorn_iter)
        loss, aux = _loss_per_ts(w, r_res_ts, betas_ts, w_prev_ts,
                                 fund_ts=fund_next_ts, **lp_kwargs)
        return loss, aux, w, scores

    return step
