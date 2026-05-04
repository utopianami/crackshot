"""Red tests for the portfolio trainer's JIT step.

Contract:
  portfolio_train_step(model, opt_state, x_ts, betas_ts, r_res_ts, w_prev_ts, mask_ts, key)
    → (new_model, new_opt_state, loss, aux)

  x_ts      : [N_ts, S_max, F]  per-timestamp features (padded)
  betas_ts  : [N_ts, S_max]     β(t) per symbol (padded 0)
  r_res_ts  : [N_ts, S_max]     forward β-residual return
  w_prev_ts : [N_ts, S_max]     previous signed weights
  mask_ts   : [N_ts, S_max]     1=valid, 0=pad

Inside the step:
  scores   = scorer(x_i) per entity  → [N_ts, S_max]
  w        = long_short_soft_topk(scores, k_long, k_short, eps)  → [N_ts, S_max]
             masked to 0 where mask==0 (pad)
  loss     = sum over N_ts of portfolio_loss(w, r_res, betas, w_prev, ...)

  Gradients must flow into scorer weights. Loss must be finite. aux must
  include per-bar mean return and mean turnover for logging.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.training.trainers.cs_mlp_portfolio_step import portfolio_train_step_fn


def test_portfolio_step_grad_and_finite():
    from src.models.architectures.cs_mlp_scorer import CrossSectionalScorerMLP
    import optax, equinox as eqx

    N_ts = 2
    S = 30  # stocks per timestamp
    F = 15
    k_long, k_short = 5, 5

    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)
    model = CrossSectionalScorerMLP(
        in_features=F, hidden_units=[32, 32], dropout=0.0,
        key=model_key, use_layernorm=False, activation="gelu",
    )
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    rng = np.random.default_rng(0)
    x_ts = rng.standard_normal((N_ts, S, F)).astype(np.float32)
    betas_ts = rng.uniform(0.3, 1.8, (N_ts, S)).astype(np.float32)
    r_res_ts = (rng.standard_normal((N_ts, S)) * 0.01).astype(np.float32)
    w_prev_ts = np.zeros((N_ts, S), dtype=np.float32)
    mask_ts = np.ones((N_ts, S), dtype=np.float32)

    step = portfolio_train_step_fn(
        optimizer,
        k_long=k_long, k_short=k_short, eps=1.0, sinkhorn_iter=50,
        lambda_beta=1.0, lambda_turn=1.0, lambda_vol=0.0,
        turnover_threshold=0.0, fee_per_side=5e-4,
    )
    new_model, new_opt, loss, aux = step(
        model, opt_state,
        jnp.array(x_ts), jnp.array(betas_ts), jnp.array(r_res_ts),
        jnp.array(w_prev_ts), jnp.array(mask_ts), data_key,
    )
    assert jnp.isfinite(loss)
    assert "mean_port_ret" in aux
    assert "mean_turnover_l1" in aux
    assert "mean_net_beta" in aux

    # Sanity: at least one weight in model changed after the step
    def tree_norm(t):
        leaves = [x for x in jax.tree_util.tree_leaves(t) if eqx.is_array(x)]
        return float(sum(jnp.sum(jnp.abs(l)) for l in leaves))

    before = tree_norm(eqx.filter(model, eqx.is_array))
    after = tree_norm(eqx.filter(new_model, eqx.is_array))
    assert abs(before - after) > 1e-6, "model weights did not update"


def test_portfolio_step_with_funding_penalty():
    """Step accepts fund_next_ts + lambda_fund; loss includes funding cost."""
    from src.models.architectures.cs_mlp_scorer import CrossSectionalScorerMLP
    import optax, equinox as eqx

    N_ts, S, F = 2, 30, 15
    k_long, k_short = 5, 5
    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)
    model = CrossSectionalScorerMLP(
        in_features=F, hidden_units=[32, 32], dropout=0.0,
        key=model_key, use_layernorm=False, activation="gelu",
    )
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    rng = np.random.default_rng(0)
    x_ts = rng.standard_normal((N_ts, S, F)).astype(np.float32)
    betas_ts = rng.uniform(0.3, 1.8, (N_ts, S)).astype(np.float32)
    r_res_ts = (rng.standard_normal((N_ts, S)) * 0.01).astype(np.float32)
    w_prev_ts = np.zeros((N_ts, S), dtype=np.float32)
    mask_ts = np.ones((N_ts, S), dtype=np.float32)
    fund_ts = (rng.standard_normal((N_ts, S)) * 0.001).astype(np.float32)

    step = portfolio_train_step_fn(
        optimizer,
        k_long=k_long, k_short=k_short, eps=1.0, sinkhorn_iter=50,
        lambda_beta=1.0, lambda_turn=1.0, lambda_vol=0.0,
        turnover_threshold=0.0, fee_per_side=5e-4,
        lambda_fund=2.0,
    )
    _, _, loss, aux = step(
        model, opt_state,
        jnp.array(x_ts), jnp.array(betas_ts), jnp.array(r_res_ts),
        jnp.array(w_prev_ts), jnp.array(mask_ts), data_key,
        fund_next_ts=jnp.array(fund_ts),
    )
    assert jnp.isfinite(loss)
    assert "mean_fund_term" in aux


def test_portfolio_step_respects_mask_pads_dont_matter():
    """Adding pad rows must not change the loss or gradient."""
    from src.models.architectures.cs_mlp_scorer import CrossSectionalScorerMLP
    import optax, equinox as eqx

    N_ts = 1
    S_small = 20
    S_pad = 30
    F = 15
    k_long, k_short = 5, 5

    key = jax.random.PRNGKey(1)
    model_key, data_key = jax.random.split(key)
    model = CrossSectionalScorerMLP(
        in_features=F, hidden_units=[16], dropout=0.0,
        key=model_key, use_layernorm=False, activation="gelu",
    )
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    rng = np.random.default_rng(2)
    x_small = rng.standard_normal((1, S_small, F)).astype(np.float32)
    betas_small = rng.uniform(0.3, 1.8, (1, S_small)).astype(np.float32)
    r_small = (rng.standard_normal((1, S_small)) * 0.01).astype(np.float32)
    w_prev_small = np.zeros((1, S_small), dtype=np.float32)
    mask_small = np.ones((1, S_small), dtype=np.float32)

    x_pad = np.zeros((1, S_pad, F), dtype=np.float32)
    x_pad[:, :S_small] = x_small
    betas_pad = np.zeros((1, S_pad), dtype=np.float32)
    betas_pad[:, :S_small] = betas_small
    r_pad = np.zeros((1, S_pad), dtype=np.float32)
    r_pad[:, :S_small] = r_small
    w_prev_pad = np.zeros((1, S_pad), dtype=np.float32)
    mask_pad = np.zeros((1, S_pad), dtype=np.float32)
    mask_pad[:, :S_small] = 1.0

    step = portfolio_train_step_fn(
        optimizer,
        k_long=k_long, k_short=k_short, eps=1.0, sinkhorn_iter=50,
        lambda_beta=1.0, lambda_turn=1.0, lambda_vol=0.0,
        turnover_threshold=0.0, fee_per_side=5e-4,
    )
    _, _, loss_a, _ = step(
        model, opt_state,
        jnp.array(x_small), jnp.array(betas_small), jnp.array(r_small),
        jnp.array(w_prev_small), jnp.array(mask_small), data_key,
    )
    _, _, loss_b, _ = step(
        model, opt_state,
        jnp.array(x_pad), jnp.array(betas_pad), jnp.array(r_pad),
        jnp.array(w_prev_pad), jnp.array(mask_pad), data_key,
    )
    # Padded rows (extreme negative scores if pushed to soft_topk directly would distort).
    # Test: masking must make the loss match to within a small tolerance.
    np.testing.assert_allclose(float(loss_a), float(loss_b), rtol=5e-2)
