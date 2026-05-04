"""Factory for portfolio scorer bodies.

Dispatches on cfg.scorer_name (mlp / tab_m / ft_t / cs_attn / resmlp).
Returned modules must honor one of two call signatures:

  per-stock scorer (default):
      model(x: jnp.ndarray[D], *, key, inference: bool) -> jnp.ndarray scalar

  cross-entity scorer (attribute ``cross_entity_scoring = True``):
      model(x: jnp.ndarray[S, D], *, key, inference: bool) -> jnp.ndarray[S]

The training step (`_scores_per_ts` in cs_mlp_portfolio_step) handles both.
"""
from __future__ import annotations

import jax

from src.models.architectures.cs_mlp_scorer import CrossSectionalScorerMLP
from src.models.architectures.portfolio.tab_m_scorer import TabMScorer
from src.models.architectures.portfolio.ft_t_scorer import FTTransformerScorer
from src.models.architectures.portfolio.cs_attn_scorer import CSAttnScorer
from src.models.architectures.portfolio.resmlp_scorer import ResMLPScorer
from src.models.architectures.portfolio.tabsyn_latent_scorer import TabsynLatentScorer
from src.models.architectures.portfolio.dcn_scorer import DCNScorer
from src.models.architectures.portfolio.mh_head_scorer import MHHeadScorer
from src.models.architectures.portfolio.sym_emb_scorer import SymEmbScorer


def build_portfolio_scorer(cfg_m, aug_dim: int, key: jax.Array, n_symbols: int | None = None):
    name = str(getattr(cfg_m, "scorer_name", "mlp")).lower()
    if name in ("mlp", "cs_mlp", ""):
        return CrossSectionalScorerMLP(
            in_features=aug_dim,
            hidden_units=list(cfg_m.hidden_units),
            dropout=float(cfg_m.dropout),
            key=key,
            use_layernorm=bool(cfg_m.use_layernorm),
            activation=str(cfg_m.activation),
        )
    if name == "tab_m":
        return TabMScorer(
            in_features=aug_dim,
            hidden_units=list(cfg_m.hidden_units),
            dropout=float(cfg_m.dropout),
            k=int(cfg_m.tab_m_k),
            activation=str(cfg_m.activation),
            use_layernorm=bool(cfg_m.use_layernorm),
            key=key,
        )
    if name == "ft_t":
        return FTTransformerScorer(
            n_features=aug_dim,
            d_token=int(cfg_m.ft_t_d_token),
            n_layers=int(cfg_m.ft_t_n_layers),
            n_heads=int(cfg_m.ft_t_n_heads),
            ffn_mult=int(cfg_m.ft_t_ffn_mult),
            dropout=float(cfg_m.dropout),
            key=key,
        )
    if name == "cs_attn":
        return CSAttnScorer(
            in_features=aug_dim,
            d_model=int(cfg_m.cs_attn_d_model),
            n_layers=int(cfg_m.cs_attn_n_layers),
            n_heads=int(cfg_m.cs_attn_n_heads),
            ffn_mult=int(cfg_m.cs_attn_ffn_mult),
            dropout=float(cfg_m.dropout),
            key=key,
        )
    if name == "resmlp":
        return ResMLPScorer(
            in_features=aug_dim,
            hidden_units=list(cfg_m.hidden_units),
            dropout=float(cfg_m.dropout),
            activation=str(cfg_m.activation),
            use_layernorm=bool(cfg_m.use_layernorm),
            key=key,
        )
    if name == "tabsyn_latent":
        return TabsynLatentScorer(
            n_features=aug_dim,
            d_token=int(cfg_m.tabsyn_d_token),
            d_latent=int(cfg_m.tabsyn_d_latent),
            n_enc_layers=int(cfg_m.tabsyn_n_enc_layers),
            n_dec_layers=int(cfg_m.tabsyn_n_dec_layers),
            n_heads=int(cfg_m.tabsyn_n_heads),
            ffn_mult=int(cfg_m.tabsyn_ffn_mult),
            dropout=float(cfg_m.dropout),
            hidden_units=list(cfg_m.hidden_units),
            kl_beta=float(cfg_m.tabsyn_kl_beta),
            key=key,
        )
    if name == "dcn":
        return DCNScorer(
            in_features=aug_dim,
            hidden_units=list(cfg_m.hidden_units),
            n_cross_layers=int(cfg_m.dcn_n_cross_layers),
            dropout=float(cfg_m.dropout),
            activation=str(cfg_m.activation),
            use_layernorm=bool(cfg_m.use_layernorm),
            key=key,
        )
    if name == "mh_mlp":
        return MHHeadScorer(
            in_features=aug_dim,
            hidden_units=list(cfg_m.hidden_units),
            dropout=float(cfg_m.dropout),
            n_heads=int(cfg_m.mh_n_heads),
            use_layernorm=bool(cfg_m.use_layernorm),
            activation=str(cfg_m.activation),
            key=key,
        )
    if name == "sym_emb":
        if n_symbols is None or n_symbols <= 0:
            raise ValueError(
                "scorer_name=sym_emb requires n_symbols>0 (passed by trainer from offsets)."
            )
        return SymEmbScorer(
            in_features=aug_dim,
            n_symbols=int(n_symbols),
            emb_dim=int(cfg_m.symbol_emb_dim),
            hidden_units=list(cfg_m.hidden_units),
            dropout=float(cfg_m.dropout),
            use_layernorm=bool(cfg_m.use_layernorm),
            activation=str(cfg_m.activation),
            key=key,
        )
    raise ValueError(
        f"Unknown scorer_name={name!r}; "
        "expected mlp|tab_m|ft_t|cs_attn|resmlp|tabsyn_latent|dcn|mh_mlp|sym_emb"
    )
