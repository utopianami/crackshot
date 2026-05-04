"""CS-MLP Portfolio trainer configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CSMLPPortfolioConfig:
    # Body
    hidden_units: list[int] = field(default_factory=lambda: [512, 512, 256])
    dropout: float = 0.3
    use_layernorm: bool = False
    activation: str = "gelu"

    # Feature augmentation
    feature_mode: str = "raw_mean_zscore"
    feature_indices: list[int] = field(default_factory=list)

    # Portfolio head (SOFT long/short top-k)
    k_long: int = 10
    k_short: int = 10
    eps_start: float = 1.0   # warmup ε (spreads gradient to all symbols)
    eps_end: float = 0.1     # final ε (sharper selection)
    eps_warmup_epochs: int = 20  # epochs to anneal ε linearly from start→end
    sinkhorn_iter: int = 100

    # Portfolio loss weights
    lambda_beta: float = 1.0
    lambda_turn: float = 1.0
    lambda_vol: float = 0.0      # single-bar only; keep 0 unless aggregating time series
    turnover_threshold: float = 0.0
    fee_per_side: float = 5e-4   # 5bp per side, matches Binance perp taker
    lambda_fund: float = 0.0     # funding-cost penalty weight; 0 disables
    fund_idx: int = -1           # label column index for fwd_funding_H; -1 = disabled

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 50
    patience: int = 10
    # SWA / EMA on model parameters (B3 experiment)
    use_ema_weights: bool = False
    ema_decay: float = 0.999
    max_stocks: int = 320        # full universe at a timestamp (298 + margin)
    n_timestamps_per_batch: int = 8   # timestamps per training step
    warmup_epochs: int = 5
    optimizer: str = "adamw"
    sgd_momentum: float = 0.9

    # Target selection (index into label_columns = [fwd_res_1h, fwd_res_4h, fwd_res_24h, beta, vol_dummy])
    target_idx: int = 1          # fwd_res_4h by default
    target_idxs: list[int] = field(default_factory=list)  # 비어있으면 single target_idx 사용. 리스트면 평균 (multi-horizon target).
    beta_idx: int = 3            # beta
    vol_idx: int = 4

    # Scorer body selection: "mlp" (default) | "tab_m" | "ft_t" | "cs_attn"
    scorer_name: str = "mlp"
    # TabM (arxiv 2410.24210): rank-1 BatchEnsemble — k parallel predictions from one forward.
    tab_m_k: int = 4               # ensemble size (1 = plain MLP, recommended 4-8)
    # FT-Transformer (arxiv 2106.11959): numerical-feature tokenization + Transformer + CLS head.
    ft_t_d_token: int = 64         # per-feature token dim
    ft_t_n_layers: int = 3
    ft_t_n_heads: int = 4
    ft_t_ffn_mult: int = 2         # FFN hidden = ffn_mult * d_token
    # cs-attn (MASTER-style, arxiv 2312.15235): cross-entity self-attention within a timestamp.
    cs_attn_d_model: int = 64
    cs_attn_n_layers: int = 2
    cs_attn_n_heads: int = 4
    cs_attn_ffn_mult: int = 2
    # Tabsyn-latent (arxiv 2310.09656, ICLR 2024): Transformer VAE encoder.
    tabsyn_d_token: int = 32
    tabsyn_d_latent: int = 16
    tabsyn_n_enc_layers: int = 2
    tabsyn_n_dec_layers: int = 2
    tabsyn_n_heads: int = 4
    tabsyn_ffn_mult: int = 2
    tabsyn_kl_beta: float = 1e-3         # KL weight on VAE aux loss
    tabsyn_aux_loss_weight: float = 0.1  # auxiliary VAE reconstruction loss multiplier
    # DCN-V2 (arxiv 2008.13535): explicit feature-cross via cross-net layers.
    dcn_n_cross_layers: int = 2
    # MH-head MLP: n parallel linear heads, scalar score = mean(heads).
    mh_n_heads: int = 3
    # Symbol embedding: per-coin learnable bias concat with augmented features.
    symbol_emb_dim: int = 32
