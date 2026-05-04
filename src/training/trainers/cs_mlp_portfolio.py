"""Cross-sectional MLP trainer with differentiable portfolio loss.

Phase 2-R: the model emits per-entity scalar scores. A SOFT long/short
top-k head projects them to signed weights. The portfolio loss rewards
β-residual return while penalising β-imbalance and turnover. Everything
is differentiable end-to-end.

Data contract (see scripts/binance/build_npy_phase2.py):
  X: [N, 15] features (12 klines/funding + 3 funding clock)
  y: [N, 5]  labels = [fwd_res_1h, fwd_res_4h, fwd_res_24h, beta, vol_dummy]
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.config.models.cs_mlp_portfolio import CSMLPPortfolioConfig
from src.models.architectures.cs_mlp_scorer import CrossSectionalScorerMLP
from src.training.checkpoint import save_best_checkpoint
from src.training.registry import BaseTrainer, register_trainer
from src.training.trainers.cs_mlp_portfolio_step import (
    portfolio_eval_step_fn,
    portfolio_train_step_fn,
)

log = logging.getLogger(__name__)


@register_trainer(
    "cs_mlp_portfolio",
    config_key="cs_mlp_portfolio",
    model_type="reg",
    default_config=asdict(CSMLPPortfolioConfig()),
)
class CSMLPPortfolioTrainer(BaseTrainer):
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.model: CrossSectionalScorerMLP | None = None
        self.feature_columns: list[str] | None = None
        self._feature_indices: np.ndarray | None = None
        self._val_predictions: np.ndarray | None = None
        self._val_actual: np.ndarray | None = None

    # Dispatch scorer construction by cfg.scorer_name (default "mlp" → CrossSectionalScorerMLP).
    # Returned module must have __call__ matching either
    #   per-stock:    (x: [D], *, key, inference) -> scalar
    #   cross-entity: (x: [S, D], *, key, inference) -> [S], with attribute cross_entity_scoring=True
    def _build_scorer(self, aug_dim: int, key: jax.Array):
        from src.training.heads.portfolio_scorer_factory import build_portfolio_scorer
        cfg_m = self.cfg.cs_mlp_portfolio
        n_symbols = int(getattr(self, "_n_symbols", 0))
        return build_portfolio_scorer(cfg_m, aug_dim, key, n_symbols=n_symbols)

    def _setup_sym_emb(
        self,
        offsets: dict,
        x_size: int,
        sym_to_id_override: dict | None = None,
    ) -> None:
        """If scorer is sym_emb, build a per-row symbol-id index. Otherwise None.

        Padding rows (slots beyond a timestamp's actual stock count) are marked
        with id == n_symbols, which the SymEmbScorer routes to a dedicated
        padding embedding row that the loss ignores via the existing mask.

        At training time we derive sym_to_id from sorted(offsets.keys()) and
        persist that map via save(). At inference we MUST reuse the trained
        map (passed as ``sym_to_id_override``) — rebuilding from a different
        offsets ordering would silently miswire the embedding lookups. Symbols
        not present in the trained map fall back to the padding id.
        """
        cfg_m = self.cfg.cs_mlp_portfolio
        is_sym_emb = str(getattr(cfg_m, "scorer_name", "mlp")).lower() == "sym_emb"
        if not is_sym_emb:
            self._sym_id_per_row = None
            self._n_symbols = 0
            self._sym_to_id = {}
            return
        if sym_to_id_override is not None:
            sym_to_id = dict(sym_to_id_override)
            self._n_symbols = len(sym_to_id)
        else:
            sym_keys = sorted(offsets.keys())
            sym_to_id = {sym: i for i, sym in enumerate(sym_keys)}
            self._n_symbols = len(sym_keys)
        self._sym_to_id = sym_to_id
        self._sym_id_per_row = np.full(x_size, self._n_symbols, dtype=np.int32)
        for sym, info in offsets.items():
            sid = sym_to_id.get(sym, self._n_symbols)  # unknown → padding id
            self._sym_id_per_row[info["start"]:info["end"]] = sid

    # ----- augmentation -------------------------------------------------
    def _augment_features(self, X_raw: np.ndarray, mode: str) -> np.ndarray:
        if mode == "raw":
            return X_raw
        market_mean = np.mean(X_raw, axis=0, keepdims=True)
        market_std = np.std(X_raw, axis=0, keepdims=True) + 1e-8
        z_score = (X_raw - market_mean) / market_std
        if mode == "zscore":
            return z_score
        if mode == "raw_zscore":
            return np.concatenate([X_raw, z_score], axis=1)
        if mode == "rankgauss":
            from scipy.special import erfinv
            n = X_raw.shape[0]
            order = np.argsort(np.argsort(X_raw, axis=0), axis=0).astype(np.float64)
            quantiles = (order + 1.0) / (n + 1.0)
            return (np.sqrt(2.0) * erfinv(2.0 * quantiles - 1.0)).astype(np.float32)
        if mode == "rmz_extended":
            # raw + mean + std + median + pct_rank + z = 6 × n_features
            n = X_raw.shape[0]
            market_median = np.median(X_raw, axis=0, keepdims=True)
            order = np.argsort(np.argsort(X_raw, axis=0), axis=0).astype(np.float32)
            pct_rank = ((order + 0.5) / max(n, 1) - 0.5).astype(np.float32)
            return np.concatenate([
                X_raw,
                np.broadcast_to(market_mean, X_raw.shape),
                np.broadcast_to(market_std, X_raw.shape),
                np.broadcast_to(market_median, X_raw.shape),
                pct_rank,
                z_score,
            ], axis=1).astype(np.float32)
        # raw_mean_zscore (default)
        market_mean_expanded = np.broadcast_to(market_mean, X_raw.shape)
        return np.concatenate([X_raw, market_mean_expanded, z_score], axis=1)

    def _aug_dim(self, n_features: int, mode: str) -> int:
        if mode in ("raw", "zscore", "rank", "rankgauss"):
            return n_features
        if mode in ("raw_zscore",):
            return n_features * 2
        if mode == "rmz_extended":
            return n_features * 6
        return n_features * 3  # raw_mean_zscore

    # ----- timestamp grouping ------------------------------------------
    def _build_ts_groups(
        self, offsets: dict, y: np.ndarray, split: str, max_stocks: int,
        target_idx: int, vol_idx: int, for_inference: bool = False,
    ):
        lengths = {
            "train": {k: v["train_end"] - v["start"] for k, v in offsets.items()},
            "val":   {k: v["val_end"] - v["train_end"] for k, v in offsets.items()},
            "test":  {k: v["end"] - v["val_end"] for k, v in offsets.items()},
        }[split]
        max_len = max(lengths.values())

        ts_to_rows: dict[int, list[int]] = {}
        for sym, info in offsets.items():
            if split == "train":
                s, e = info["start"], info["train_end"]
            elif split == "val":
                s, e = info["train_end"], info["val_end"]
            else:
                s, e = info["val_end"], info["end"]
            stock_len = e - s
            offset = max_len - stock_len  # align to end
            for t in range(stock_len):
                rel_t = offset + t
                abs_row = s + t
                vol_val = float(y[abs_row, vol_idx])
                tgt_val = float(y[abs_row, target_idx])
                if vol_val != vol_val or tgt_val != tgt_val:
                    continue
                if not for_inference and (vol_val == 0 or tgt_val == 0):
                    continue
                ts_to_rows.setdefault(rel_t, []).append(abs_row)

        sorted_ts = sorted(ts_to_rows.keys())
        n_ts = len(sorted_ts)
        ts_rows = np.zeros((n_ts, max_stocks), dtype=np.int64)
        ts_counts = np.zeros(n_ts, dtype=np.int32)
        for i, ts in enumerate(sorted_ts):
            rows = ts_to_rows[ts]
            n = min(len(rows), max_stocks)
            ts_rows[i, :n] = rows[:n]
            ts_counts[i] = n
        log.info(
            f"Timestamp groups ({split}): {n_ts} timestamps, stocks/ts: "
            f"min={int(ts_counts.min())} max={int(ts_counts.max())} mean={ts_counts.mean():.0f}"
        )
        return ts_rows, ts_counts, n_ts

    # ----- batch loader ------------------------------------------------
    def _load_batch(self, X, y, beta_col, r_col, ts_rows, ts_counts, ts_idx, max_stocks):
        n_ts = len(ts_idx)
        F = len(self._feature_indices) if self._feature_indices is not None else X.shape[1]
        aug = self._aug_dim(F, self._feature_mode)
        sym_active = getattr(self, "_sym_id_per_row", None) is not None
        total_dim = aug + (1 if sym_active else 0)

        x_b = np.zeros((n_ts, max_stocks, total_dim), dtype=np.float32)
        if sym_active:
            # Pre-fill the sym-id column for ALL slots with the padding id so
            # masked-out slots map to the padding embedding (not a real symbol).
            x_b[:, :, -1] = float(self._n_symbols)
        beta_b = np.zeros((n_ts, max_stocks), dtype=np.float32)
        r_b = np.zeros((n_ts, max_stocks), dtype=np.float32)
        mask_b = np.zeros((n_ts, max_stocks), dtype=np.float32)
        fund_b = np.zeros((n_ts, max_stocks), dtype=np.float32) if self._fund_idx >= 0 else None

        valid = []
        for i, ts_i in enumerate(ts_idx):
            n = int(ts_counts[ts_i])
            if n < 2:
                continue
            n_stocks = min(n, max_stocks)
            rows = ts_rows[ts_i, :n_stocks]
            x_raw = np.asarray(X[rows], dtype=np.float32)
            if self._feature_indices is not None:
                x_raw = x_raw[:, self._feature_indices]
            x_aug = self._augment_features(x_raw, self._feature_mode)
            x_aug = np.nan_to_num(x_aug, nan=0.0)
            x_b[i, :n_stocks, :aug] = x_aug
            if sym_active:
                x_b[i, :n_stocks, -1] = self._sym_id_per_row[rows].astype(np.float32)

            ys = np.asarray(y[rows], dtype=np.float32)
            beta_b[i, :n_stocks] = np.nan_to_num(ys[:, self._beta_idx], nan=0.0)
            if self._target_idxs:
                # multi-horizon target: 평균
                tgt_arr = np.nan_to_num(ys[:, self._target_idxs], nan=0.0).mean(axis=1)
                r_b[i, :n_stocks] = tgt_arr
            else:
                r_b[i, :n_stocks] = np.nan_to_num(ys[:, self._target_idx], nan=0.0)
            mask_b[i, :n_stocks] = 1.0
            if fund_b is not None:
                fund_b[i, :n_stocks] = np.nan_to_num(ys[:, self._fund_idx], nan=0.0)
            valid.append(i)

        if not valid:
            return None
        return x_b, beta_b, r_b, mask_b, fund_b

    # ----- fit ---------------------------------------------------------
    def fit_from_npy(self, npy_dir: str | Path) -> None:
        npy_dir = Path(npy_dir)
        cfg_m: CSMLPPortfolioConfig = self.cfg.cs_mlp_portfolio

        with open(npy_dir / "meta.json") as f:
            meta = json.load(f)
        self.feature_columns = meta["feature_columns"]
        n_features_in = meta["n_features"]
        log.info(f"Features loaded: {self.feature_columns}")
        log.info(f"Labels: {meta['label_columns']}")

        self._target_idx = int(cfg_m.target_idx)
        self._target_idxs = list(cfg_m.target_idxs) if cfg_m.target_idxs else []
        if self._target_idxs:
            log.info(f"[multi-horizon target] using avg of label indices {self._target_idxs}")
        self._beta_idx = int(cfg_m.beta_idx)
        self._vol_idx = int(cfg_m.vol_idx)
        self._fund_idx = int(getattr(cfg_m, "fund_idx", -1))
        self._lambda_fund = float(getattr(cfg_m, "lambda_fund", 0.0))
        self._feature_mode = str(cfg_m.feature_mode)
        if self._fund_idx >= 0 and self._lambda_fund > 0:
            n_labels = len(meta["label_columns"])
            if self._fund_idx >= n_labels:
                raise ValueError(
                    f"fund_idx={self._fund_idx} out of range for label_columns "
                    f"(len={n_labels}). Rebuild npy with fwd_funding_H label."
                )
            log.info(f"[CS-MLP-P] funding loss enabled: lambda_fund={self._lambda_fund} "
                     f"fund_idx={self._fund_idx} ({meta['label_columns'][self._fund_idx]})")

        feat_idx = list(cfg_m.feature_indices) if cfg_m.feature_indices else []
        self._feature_indices = np.asarray(feat_idx, dtype=int) if feat_idx else None
        if self._feature_indices is not None:
            n_features_in = len(self._feature_indices)
        aug_dim = self._aug_dim(n_features_in, self._feature_mode)
        log.info(
            f"[CS-MLP-Portfolio] hidden={cfg_m.hidden_units} in={n_features_in} "
            f"aug={aug_dim} mode={self._feature_mode}"
        )

        X = np.load(npy_dir / "X.npy", mmap_mode="r")
        y = np.load(npy_dir / "y.npy", mmap_mode="r")
        from src.data.offsets_override import maybe_patch_offsets_from_cfg
        with open(npy_dir / "offsets.json") as f:
            offsets = json.load(f)
        offsets = maybe_patch_offsets_from_cfg(offsets, npy_dir, self.cfg.data)

        # silent-NaN guard: abort if NPY has NaN before training runs (downstream
        # nan_to_num would mask it). protects against build-time data drift.
        if np.isnan(X).any():
            n_nan = int(np.isnan(X).sum())
            raise ValueError(
                f"[silent-NaN guard] X.npy contains {n_nan} NaN values. "
                "Rebuild NPY or investigate before training. "
                "(See feedback_no_silent_nan_imputation.md.)"
            )
        if np.isnan(y).any():
            n_nan = int(np.isnan(y).sum())
            raise ValueError(f"[silent-NaN guard] y.npy contains {n_nan} NaN values.")

        # Symbol-embedding setup (no-op unless scorer_name == "sym_emb").
        # Builds self._sym_id_per_row keyed by abs row index, used by _load_batch.
        # Must precede _build_scorer because the factory needs n_symbols.
        self._setup_sym_emb(offsets, x_size=int(X.shape[0]))

        ts_rows_tr, ts_counts_tr, n_ts_tr = self._build_ts_groups(
            offsets, y, "train", cfg_m.max_stocks, self._target_idx, self._vol_idx,
        )
        ts_rows_va, ts_counts_va, n_ts_va = self._build_ts_groups(
            offsets, y, "val", cfg_m.max_stocks, self._target_idx, self._vol_idx,
        )

        key = jax.random.PRNGKey(self.cfg.seed)
        key, mkey = jax.random.split(key)
        self.model = self._build_scorer(aug_dim, mkey)

        n_ts_per_batch = int(cfg_m.n_timestamps_per_batch)
        steps_per_epoch = max(1, n_ts_tr // n_ts_per_batch)
        total_steps = max(1, int(cfg_m.epochs) * steps_per_epoch)
        warmup_steps = min(total_steps - 1, int(cfg_m.warmup_epochs) * steps_per_epoch)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=cfg_m.lr * 0.01,
            peak_value=cfg_m.lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=cfg_m.lr * 0.01,
        )
        optimizer = optax.adamw(schedule, weight_decay=cfg_m.weight_decay)
        opt_state = optimizer.init(eqx.filter(self.model, eqx.is_array))

        best_val = float("inf")
        patience = 0
        best_model = self.model

        # B3: EMA shadow on model parameters (SWA-style)
        ema_model = self.model if cfg_m.use_ema_weights else None
        ema_decay = float(cfg_m.ema_decay)

        # ε anneal schedule — linear from eps_start → eps_end over eps_warmup_epochs
        def eps_for_epoch(ep):
            if cfg_m.eps_warmup_epochs <= 0:
                return cfg_m.eps_end
            frac = min(1.0, ep / cfg_m.eps_warmup_epochs)
            return cfg_m.eps_start + frac * (cfg_m.eps_end - cfg_m.eps_start)

        # w_prev: one rolling per timestamp bucket. At start, zeros. Updated
        # across steps by using the most recent w from that ts-idx.
        # For simplicity, use zeros at every step (turnover proxy = |w|_1).
        zeros_prev = np.zeros((n_ts_per_batch * 2, cfg_m.max_stocks), dtype=np.float32)

        for epoch in range(int(cfg_m.epochs)):
            eps_curr = eps_for_epoch(epoch)
            step_fn = portfolio_train_step_fn(
                optimizer,
                k_long=int(cfg_m.k_long), k_short=int(cfg_m.k_short),
                eps=float(eps_curr), sinkhorn_iter=int(cfg_m.sinkhorn_iter),
                lambda_beta=float(cfg_m.lambda_beta),
                lambda_turn=float(cfg_m.lambda_turn),
                lambda_vol=float(cfg_m.lambda_vol),
                turnover_threshold=float(cfg_m.turnover_threshold),
                fee_per_side=float(cfg_m.fee_per_side),
                lambda_fund=self._lambda_fund,
                lambda_aux=float(getattr(cfg_m, "tabsyn_aux_loss_weight", 0.0))
                           if str(getattr(cfg_m, "scorer_name", "mlp")) == "tabsyn_latent" else 0.0,
            )
            eval_fn = portfolio_eval_step_fn(
                k_long=int(cfg_m.k_long), k_short=int(cfg_m.k_short),
                eps=float(eps_curr), sinkhorn_iter=int(cfg_m.sinkhorn_iter),
                lambda_beta=float(cfg_m.lambda_beta),
                lambda_turn=float(cfg_m.lambda_turn),
                lambda_vol=float(cfg_m.lambda_vol),
                turnover_threshold=float(cfg_m.turnover_threshold),
                fee_per_side=float(cfg_m.fee_per_side),
                lambda_fund=self._lambda_fund,
            )

            rng = np.random.default_rng(self.cfg.seed + epoch)
            ts_perm = rng.permutation(n_ts_tr)
            epoch_loss = 0.0
            n_batches = 0
            tr_ret_sum = 0.0
            tr_turn_sum = 0.0

            for ts_start in range(0, n_ts_tr, n_ts_per_batch):
                ts_idx = ts_perm[ts_start:ts_start + n_ts_per_batch]
                batch = self._load_batch(X, y, self._beta_idx, self._target_idx,
                                          ts_rows_tr, ts_counts_tr, ts_idx, cfg_m.max_stocks)
                if batch is None:
                    continue
                x_b, beta_b, r_b, mask_b, fund_b = batch
                key, sub = jax.random.split(key)
                fund_arg = jnp.asarray(fund_b) if fund_b is not None else None
                self.model, opt_state, loss, aux = step_fn(
                    self.model, opt_state,
                    jnp.asarray(x_b), jnp.asarray(beta_b), jnp.asarray(r_b),
                    jnp.asarray(zeros_prev[:len(ts_idx)]),
                    jnp.asarray(mask_b), sub,
                    fund_next_ts=fund_arg,
                )
                # B3: EMA shadow update — only on inexact-array (float) leaves
                if ema_model is not None:
                    ema_arr, ema_static = eqx.partition(ema_model, eqx.is_inexact_array)
                    cur_arr, _ = eqx.partition(self.model, eqx.is_inexact_array)
                    new_arr = jax.tree_util.tree_map(
                        lambda e, m: ema_decay * e + (1.0 - ema_decay) * m, ema_arr, cur_arr
                    )
                    ema_model = eqx.combine(new_arr, ema_static)
                epoch_loss += float(loss)
                tr_ret_sum += float(aux["mean_port_ret"])
                tr_turn_sum += float(aux["mean_turnover_l1"])
                n_batches += 1

            # Validation
            val_loss_sum = 0.0
            val_ret_sum = 0.0
            val_turn_sum = 0.0
            val_batches = 0
            for ts_start in range(0, n_ts_va, n_ts_per_batch * 2):
                ts_idx = np.arange(ts_start, min(ts_start + n_ts_per_batch * 2, n_ts_va))
                batch = self._load_batch(X, y, self._beta_idx, self._target_idx,
                                          ts_rows_va, ts_counts_va, ts_idx, cfg_m.max_stocks)
                if batch is None:
                    continue
                x_b, beta_b, r_b, mask_b, fund_b = batch
                key, sub = jax.random.split(key)
                fund_arg = jnp.asarray(fund_b) if fund_b is not None else None
                vl, vaux, _, _ = eval_fn(
                    self.model,
                    jnp.asarray(x_b), jnp.asarray(beta_b), jnp.asarray(r_b),
                    jnp.asarray(zeros_prev[:len(ts_idx)]),
                    jnp.asarray(mask_b), sub,
                    fund_next_ts=fund_arg,
                )
                val_loss_sum += float(vl)
                val_ret_sum += float(vaux["mean_port_ret"])
                val_turn_sum += float(vaux["mean_turnover_l1"])
                val_batches += 1

            tr_loss = epoch_loss / max(n_batches, 1)
            val_loss = val_loss_sum / max(val_batches, 1)
            val_ret = val_ret_sum / max(val_batches, 1)
            val_turn = val_turn_sum / max(val_batches, 1)

            log.info(
                f"[CS-MLP-P] ep={epoch:3d} eps={eps_curr:.3f} "
                f"tr_loss={tr_loss:.5f} val_loss={val_loss:.5f} "
                f"val_port_ret={val_ret:+.5f} val_turn_l1={val_turn:.2f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_model = self.model
                save_best_checkpoint(
                    trainer=self,
                    save_path=self._checkpoint_save_path,
                    best_model=best_model,
                    epoch=epoch,
                    val_loss=float(best_val),
                    extra_meta={"val_port_ret": float(val_ret),
                                 "val_turnover_l1": float(val_turn),
                                 "eps": float(eps_curr)},
                    logger=log,
                )
                patience = 0
            else:
                patience += 1
                if patience >= int(cfg_m.patience):
                    log.info(f"[CS-MLP-P] Early stopping at ep={epoch}. best_val={best_val:.5f}")
                    break

        # B3: if EMA enabled, replace final model with EMA shadow
        if ema_model is not None:
            log.info(f"[CS-MLP-P] B3 EMA: replacing best_model with ema_model (decay={ema_decay})")
            # Save EMA as final checkpoint (overrides best)
            save_best_checkpoint(
                trainer=self,
                save_path=self._checkpoint_save_path,
                best_model=ema_model,
                epoch=int(cfg_m.epochs),
                val_loss=float(best_val),
                extra_meta={"ema_decay": ema_decay, "swa_replacement": True},
                logger=log,
            )
            self.model = ema_model
        else:
            self.model = best_model
        # cache val predictions (scores only — weights computed at inference)
        self._val_predictions, self._val_actual = self._score_split(
            X, y, ts_rows_va, ts_counts_va, n_ts_va, cfg_m.max_stocks,
        )

    def _score_split(self, X, y, ts_rows, ts_counts, n_ts, max_stocks):
        assert self.model is not None
        key = jax.random.PRNGKey(0)

        is_ce = bool(getattr(self.model, "cross_entity_scoring", False))

        @eqx.filter_jit
        def pred_batch_per_stock(model, x, key):
            def per_stock(xi, ki):
                return model(xi, key=ki, inference=True)
            keys = jax.random.split(key, x.shape[0])
            return jax.vmap(per_stock)(x, keys)

        @eqx.filter_jit
        def pred_batch_ce(model, x, key):
            return model(x, key=key, inference=True)

        pred_batch = pred_batch_ce if is_ce else pred_batch_per_stock

        sym_active = getattr(self, "_sym_id_per_row", None) is not None
        scores_map: dict[int, float] = {}
        actual_map: dict[int, float] = {}
        for ts_i in range(n_ts):
            n = int(ts_counts[ts_i])
            if n < 2:
                continue
            rows = ts_rows[ts_i, :n]
            x_raw = np.asarray(X[rows], dtype=np.float32)
            if self._feature_indices is not None:
                x_raw = x_raw[:, self._feature_indices]
            x_aug = np.nan_to_num(
                self._augment_features(x_raw, self._feature_mode), nan=0.0,
            )
            if sym_active:
                sym_col = self._sym_id_per_row[rows].astype(np.float32)[:, None]
                x_aug = np.concatenate([x_aug, sym_col], axis=1)
            key, sub = jax.random.split(key)
            scores = np.asarray(pred_batch(self.model, jnp.asarray(x_aug), sub))
            y_batch = np.asarray(y[rows])
            for j, row_idx in enumerate(rows):
                scores_map[int(row_idx)] = float(scores[j])
                actual_map[int(row_idx)] = float(y_batch[j, self._target_idx])

        sorted_rows = sorted(scores_map.keys())
        all_scores = np.array([scores_map[r] for r in sorted_rows], dtype=np.float32)
        all_actuals = np.array([actual_map[r] for r in sorted_rows], dtype=np.float32)
        # NaN/Inf guard — silent NaN in scores would poison portfolio weights
        # via softmax/top-k and is almost impossible to debug downstream.
        n_nan = int(np.isnan(all_scores).sum())
        n_inf = int(np.isinf(all_scores).sum())
        assert n_nan == 0 and n_inf == 0, (
            f"CS-MLP-P score contains NaN/Inf: nan={n_nan} inf={n_inf} out of {all_scores.size}. "
            "Likely feature NaN surviving augmentation, or exploded weights."
        )
        return all_scores, all_actuals

    def predict_npy(self, npy_dir: str | Path, split: str = "test"):
        assert self.model is not None
        npy_dir = Path(npy_dir)
        cfg_m = self.cfg.cs_mlp_portfolio
        X = np.load(npy_dir / "X.npy", mmap_mode="r")
        y = np.load(npy_dir / "y.npy", mmap_mode="r")
        from src.data.offsets_override import maybe_patch_offsets_from_cfg
        from src.training.ts_group_guard import (
            build_abs_to_ts,
            assert_single_ts_per_rel_t,
            build_ts_groups_wallclock,
        )
        with open(npy_dir / "offsets.json") as f:
            offsets_raw = json.load(f)
        orig_offsets = {s: dict(v) for s, v in offsets_raw.items()}
        offsets = maybe_patch_offsets_from_cfg(offsets_raw, npy_dir, self.cfg.data)

        # Rebuild sym_id_per_row at inference. If load() restored a saved
        # sym_to_id (from cs_mlp_portfolio_config.json), reuse it so embedding
        # ids stay locked to training-time mapping; otherwise fall back to the
        # current offsets order (works for fresh fits and self-window apply).
        saved_map = getattr(self, "_sym_to_id_saved", None) or None
        self._setup_sym_emb(
            offsets,
            x_size=int(X.shape[0]),
            sym_to_id_override=saved_map,
        )

        # For the test split: use canonical wall-clock timestamp grouping. The
        # row-based rel_t right-align breaks when symbols have mid-window gap
        # bars — groups silently mix different wall-clocks across symbols.
        # For train/val: keep row-based grouping (no per-row timestamp metadata).
        if split == "test":
            ts_rows, ts_counts, n_ts, abs_to_ts = build_ts_groups_wallclock(
                offsets, orig_offsets, y, npy_dir, split,
                int(cfg_m.max_stocks), self._target_idx, self._vol_idx,
                for_inference=True,
            )
            # Defensive assert: every group should be internally consistent by construction.
            ts_to_rows_check = {
                t: [int(ts_rows[t, s]) for s in range(int(ts_counts[t]))]
                for t in range(n_ts)
            }
            assert_single_ts_per_rel_t(ts_to_rows_check, abs_to_ts)
            log.info(
                f"[CS-MLP-P] wallclock groups (test): n_ts={n_ts} "
                f"stocks/ts min={int(ts_counts.min()) if n_ts else 0} "
                f"max={int(ts_counts.max()) if n_ts else 0} "
                f"mean={float(ts_counts.mean()) if n_ts else 0:.1f}"
            )
        else:
            ts_rows, ts_counts, n_ts = self._build_ts_groups(
                offsets, y, split, int(cfg_m.max_stocks),
                self._target_idx, self._vol_idx, for_inference=True,
            )

        return self._score_split(X, y, ts_rows, ts_counts, n_ts, int(cfg_m.max_stocks))

    def save(self, path: str | Path) -> None:
        assert self.model is not None
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(str(path / "cs_mlp_portfolio_model.eqx"), self.model)
        with open(path / "feature_columns.json", "w") as f:
            json.dump(self.feature_columns, f)
        cfg_m = self.cfg.cs_mlp_portfolio
        with open(path / "cs_mlp_portfolio_config.json", "w") as f:
            json.dump({
                "in_features": len(self.feature_columns) if self.feature_columns else 0,
                "hidden_units": list(cfg_m.hidden_units),
                "dropout": float(cfg_m.dropout),
                "use_layernorm": bool(cfg_m.use_layernorm),
                "activation": str(cfg_m.activation),
                "feature_mode": self._feature_mode,
                "feature_indices": (self._feature_indices.tolist()
                                     if self._feature_indices is not None else None),
                "target_idx": int(self._target_idx),
                "beta_idx": int(self._beta_idx),
                "vol_idx": int(self._vol_idx),
                "k_long": int(cfg_m.k_long),
                "k_short": int(cfg_m.k_short),
                "eps_end": float(cfg_m.eps_end),
                "sinkhorn_iter": int(cfg_m.sinkhorn_iter),
                "scorer_name": str(getattr(cfg_m, "scorer_name", "mlp")),
                "tab_m_k": int(getattr(cfg_m, "tab_m_k", 4)),
                "ft_t_d_token": int(getattr(cfg_m, "ft_t_d_token", 64)),
                "ft_t_n_layers": int(getattr(cfg_m, "ft_t_n_layers", 3)),
                "ft_t_n_heads": int(getattr(cfg_m, "ft_t_n_heads", 4)),
                "ft_t_ffn_mult": int(getattr(cfg_m, "ft_t_ffn_mult", 2)),
                "cs_attn_d_model": int(getattr(cfg_m, "cs_attn_d_model", 64)),
                "cs_attn_n_layers": int(getattr(cfg_m, "cs_attn_n_layers", 2)),
                "cs_attn_n_heads": int(getattr(cfg_m, "cs_attn_n_heads", 4)),
                "cs_attn_ffn_mult": int(getattr(cfg_m, "cs_attn_ffn_mult", 2)),
                "tabsyn_d_token": int(getattr(cfg_m, "tabsyn_d_token", 32)),
                "tabsyn_d_latent": int(getattr(cfg_m, "tabsyn_d_latent", 16)),
                "tabsyn_n_enc_layers": int(getattr(cfg_m, "tabsyn_n_enc_layers", 2)),
                "tabsyn_n_dec_layers": int(getattr(cfg_m, "tabsyn_n_dec_layers", 2)),
                "tabsyn_n_heads": int(getattr(cfg_m, "tabsyn_n_heads", 4)),
                "tabsyn_ffn_mult": int(getattr(cfg_m, "tabsyn_ffn_mult", 2)),
                "tabsyn_kl_beta": float(getattr(cfg_m, "tabsyn_kl_beta", 1e-3)),
                "tabsyn_aux_loss_weight": float(getattr(cfg_m, "tabsyn_aux_loss_weight", 0.1)),
                "dcn_n_cross_layers": int(getattr(cfg_m, "dcn_n_cross_layers", 2)),
                "mh_n_heads": int(getattr(cfg_m, "mh_n_heads", 3)),
                "symbol_emb_dim": int(getattr(cfg_m, "symbol_emb_dim", 32)),
                "n_symbols": int(getattr(self, "_n_symbols", 0)),
                # Persist sym_to_id so inference rebuilds the SAME mapping even
                # if symbols are added/removed/reordered between train and apply.
                "sym_to_id": dict(getattr(self, "_sym_to_id", {})),
            }, f)
        log.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path / "feature_columns.json") as f:
            self.feature_columns = json.load(f)
        with open(path / "cs_mlp_portfolio_config.json") as f:
            cfg = json.load(f)
        self._feature_mode = cfg.get("feature_mode", "raw_mean_zscore")
        self._target_idx = int(cfg["target_idx"])
        self._beta_idx = int(cfg["beta_idx"])
        self._vol_idx = int(cfg["vol_idx"])
        fi = cfg.get("feature_indices")
        self._feature_indices = np.asarray(fi, dtype=int) if fi else None
        # Restore n_symbols *before* _build_scorer so the sym_emb factory branch
        # can size its embedding table correctly.
        self._n_symbols = int(cfg.get("n_symbols", 0))
        self._sym_id_per_row = None  # rebuilt at predict_npy from offsets.json
        # Stash the saved sym_to_id mapping for predict_npy to re-use.
        self._sym_to_id_saved = dict(cfg.get("sym_to_id", {}))
        raw_features = cfg["in_features"] if self._feature_indices is None else len(self._feature_indices)
        aug_dim = self._aug_dim(raw_features, self._feature_mode)
        # Force self.cfg.cs_mlp_portfolio to match the saved checkpoint config
        # before _build_scorer reads it. Without this patch the factory would
        # silently build the wrong scorer if the live YAML's scorer_name (or
        # any per-scorer hyperparam) drifted from training time.
        cfg_m = self.cfg.cs_mlp_portfolio
        for k in (
            "scorer_name", "tab_m_k",
            "ft_t_d_token", "ft_t_n_layers", "ft_t_n_heads", "ft_t_ffn_mult",
            "cs_attn_d_model", "cs_attn_n_layers", "cs_attn_n_heads", "cs_attn_ffn_mult",
            "tabsyn_d_token", "tabsyn_d_latent", "tabsyn_n_enc_layers",
            "tabsyn_n_dec_layers", "tabsyn_n_heads", "tabsyn_ffn_mult", "tabsyn_kl_beta",
            "dcn_n_cross_layers", "mh_n_heads", "symbol_emb_dim",
            "hidden_units", "dropout", "use_layernorm", "activation",
        ):
            if k in cfg:
                setattr(cfg_m, k, cfg[k])
        key = jax.random.PRNGKey(0)
        self.model = self._build_scorer(aug_dim, key)
        self.model = eqx.tree_deserialise_leaves(
            str(path / "cs_mlp_portfolio_model.eqx"), self.model,
        )
        log.info(f"Model loaded from {path}")
