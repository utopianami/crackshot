"""Cross-sectional MLP trainer.

Trains per-timestamp: loads all stocks for each timestamp,
computes market-level statistics, augments features with
[raw, market_mean, z_score], then classifies each stock.
"""

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

from src.config.models.cs_mlp import CSMLPConfig
from src.data.labels import to_classes
from src.models.architectures.cs_mlp import CrossSectionalMLP
from src.training.losses import label_smoothing_ce, listnet_loss
from src.training.registry import BaseTrainer, register_trainer

log = logging.getLogger(__name__)


@register_trainer("cs_mlp", config_key="cs_mlp", model_type="cls", default_config=asdict(CSMLPConfig()))
class CSMLPTrainer(BaseTrainer):
    """Cross-sectional MLP trainer — timestamp-grouped training."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        ds = cfg.dataset
        self._h0, self._h1, self._h2 = ds.horizon_indices
        self._vol_idx = ds.vol_idx
        self._num_horizons = len(ds.horizon_indices)
        self._class_interior_bins = np.array(ds.class_bins, dtype=np.float64)
        self._num_classes = len(ds.class_bins) + 1
        self.model: CrossSectionalMLP | None = None
        self.feature_columns: list[str] | None = None
        self._feature_indices: np.ndarray | None = None
        self._val_predictions: np.ndarray | None = None
        self._val_actual: np.ndarray | None = None

    def fit_from_npy(self, npy_dir: str | Path) -> None:
        npy_dir = Path(npy_dir)
        cs_cfg = self.cfg.cs_mlp

        # Load data
        in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices = \
            self._load_npy_data(npy_dir)

        # Build timestamp groups
        with open(npy_dir / "offsets.json") as f:
            offsets_dict = json.load(f)

        ts_rows_tr, ts_counts_tr, n_ts_tr = self._build_timestamp_groups(
            offsets_dict, y_train, "train", cs_cfg.max_stocks
        )
        ts_rows_va, ts_counts_va, n_ts_va = self._build_timestamp_groups(
            offsets_dict, y_train, "val", cs_cfg.max_stocks
        )

        feature_mode = getattr(cs_cfg, 'feature_mode', 'raw_mean_zscore')
        self._feature_mode = feature_mode
        self._use_layernorm = getattr(cs_cfg, 'use_layernorm', False)
        self._activation = getattr(cs_cfg, 'activation', 'gelu')
        feat_idx = getattr(cs_cfg, 'feature_indices', [])
        self._feature_indices = np.array(feat_idx, dtype=int) if feat_idx else None
        if self._feature_indices is not None:
            in_features = len(self._feature_indices)
            log.info(f"Feature selection: {len(self._feature_indices)}/{X_train.shape[1]} features")
        aug_dim = self._aug_dim(in_features, feature_mode)
        log.info(f"=== CS-MLP Training (hidden={cs_cfg.hidden_units}, "
                 f"in={in_features}, aug={aug_dim}, mode={feature_mode}) ===")

        # Create model
        key = jax.random.PRNGKey(self.cfg.seed)
        key, model_key = jax.random.split(key)
        self.model = CrossSectionalMLP(
            in_features=aug_dim,
            hidden_units=list(cs_cfg.hidden_units),
            dropout=cs_cfg.dropout,
            num_classes=self._num_classes,
            num_horizons=self._num_horizons,
            key=model_key,
            use_layernorm=cs_cfg.use_layernorm,
            activation=getattr(cs_cfg, 'activation', 'gelu'),
            residual=getattr(cs_cfg, 'residual', False),
        )

        # Train
        self.model = self._train(
            self.model, X_train, y_train, X_val, y_val,
            ts_rows_tr, ts_counts_tr, n_ts_tr,
            ts_rows_va, ts_counts_va, n_ts_va,
            cs_cfg, key,
        )

        # Val predictions
        log.info("Computing val predictions...")
        self._val_predictions, self._val_actual = self._predict_timestamps(
            X_val, y_val, ts_rows_va, ts_counts_va, n_ts_va,
        )

    def _build_timestamp_groups(
        self, offsets: dict, y: np.ndarray, split: str, max_stocks: int,
    ):
        """Build timestamp → row_indices mapping for cross-sectional training.

        Same logic as TabM trainer: relative_pos aligns timestamps across stocks.
        """
        if split == "train":
            lengths = {k: v["train_end"] - v["start"] for k, v in offsets.items()}
        elif split == "val":
            lengths = {k: v["val_end"] - v["train_end"] for k, v in offsets.items()}
        else:  # test
            lengths = {k: v["end"] - v["val_end"] for k, v in offsets.items()}
        max_len = max(lengths.values())

        ts_to_rows: dict[int, list[int]] = {}
        for stock_id, info in offsets.items():
            if split == "train":
                start, end = info["start"], info["train_end"]
            elif split == "val":
                start, end = info["train_end"], info["val_end"]
            else:  # test
                start, end = info["val_end"], info["end"]
            stock_len = end - start
            offset = max_len - stock_len  # align to end

            for t in range(stock_len):
                rel_t = offset + t
                abs_row = start + t
                # Filter: vol > 0 and target != 0
                vol_val = float(y[abs_row, self._vol_idx])
                target_val = float(y[abs_row, self._h1])
                if vol_val == 0 or target_val == 0:
                    continue
                if rel_t not in ts_to_rows:
                    ts_to_rows[rel_t] = []
                ts_to_rows[rel_t].append(abs_row)

        n_timestamps = len(ts_to_rows)
        sorted_ts = sorted(ts_to_rows.keys())
        ts_rows = np.zeros((n_timestamps, max_stocks), dtype=np.int64)
        ts_counts = np.zeros(n_timestamps, dtype=np.int32)

        for i, ts in enumerate(sorted_ts):
            rows = ts_to_rows[ts]
            n = min(len(rows), max_stocks)
            ts_rows[i, :n] = rows[:n]
            ts_counts[i] = n

        log.info(f"Timestamp groups ({split}): {n_timestamps} timestamps, "
                 f"stocks/ts: min={ts_counts.min()}, max={ts_counts.max()}, "
                 f"mean={ts_counts.mean():.0f}")
        return ts_rows, ts_counts, n_timestamps

    def _augment_features(self, X_raw: np.ndarray, feature_mode: str) -> np.ndarray:
        """Cross-sectional feature augmentation."""
        market_mean = np.mean(X_raw, axis=0, keepdims=True)
        market_std = np.std(X_raw, axis=0, keepdims=True) + 1e-8
        z_score = (X_raw - market_mean) / market_std

        if feature_mode == "zscore":
            return z_score
        elif feature_mode == "raw_zscore":
            return np.concatenate([X_raw, z_score], axis=1)
        elif feature_mode == "rank":
            from scipy.stats import rankdata
            ranks = np.apply_along_axis(
                lambda col: rankdata(col, method='average') / len(col) - 0.5,
                axis=0, arr=X_raw,
            )
            return ranks.astype(np.float32)
        else:  # raw_mean_zscore (default)
            market_mean_expanded = np.broadcast_to(market_mean, X_raw.shape)
            return np.concatenate([X_raw, market_mean_expanded, z_score], axis=1)

    def _aug_dim(self, n_features: int, feature_mode: str) -> int:
        if feature_mode in ("zscore", "rank"):
            return n_features
        elif feature_mode == "raw_zscore":
            return n_features * 2
        return n_features * 3  # raw_mean_zscore

    def _load_ts_batch(self, X, y, ts_rows, ts_counts, ts_indices, max_stocks: int):
        """Load a batch of timestamps with cross-sectional augmentation.

        Returns padded arrays for JIT compatibility:
            X_aug: [N_ts, max_stocks, aug_dim] padded augmented features
            y120: [N_ts, max_stocks] class labels (padded with 0)
            ret120: [N_ts, max_stocks] raw returns (padded with 0)
            mask: [N_ts, max_stocks] 1=valid, 0=padding
        """
        bins = self._class_interior_bins
        n_features = len(self._feature_indices) if self._feature_indices is not None else X.shape[1]
        feature_mode = getattr(self, '_feature_mode', 'raw_mean_zscore')
        aug_dim = self._aug_dim(n_features, feature_mode)

        valid_ts = []
        for ts_idx in ts_indices:
            n = int(ts_counts[ts_idx])
            if n >= 2:
                valid_ts.append(ts_idx)

        if not valid_ts:
            return None, None, None, None

        n_ts = len(valid_ts)
        X_aug_all = np.zeros((n_ts, max_stocks, aug_dim), dtype=np.float32)
        y120_all = np.zeros((n_ts, max_stocks), dtype=np.int32)
        ret120_all = np.zeros((n_ts, max_stocks), dtype=np.float32)
        mask_all = np.zeros((n_ts, max_stocks), dtype=np.float32)

        for i, ts_idx in enumerate(valid_ts):
            n_stocks = min(int(ts_counts[ts_idx]), max_stocks)
            rows = ts_rows[ts_idx, :n_stocks]

            X_raw = np.array(X[rows]).astype(np.float32)  # [S, F]
            if self._feature_indices is not None:
                X_raw = X_raw[:, self._feature_indices]
            y_batch = np.array(y[rows])
            y120 = to_classes(y_batch[:, self._h1], bins)
            ret120 = y_batch[:, self._h1].astype(np.float32)

            # Cross-sectional augmentation
            X_aug = self._augment_features(X_raw, feature_mode)
            X_aug = np.nan_to_num(X_aug, nan=0.0)

            X_aug_all[i, :n_stocks] = X_aug
            y120_all[i, :n_stocks] = y120
            ret120_all[i, :n_stocks] = ret120
            mask_all[i, :n_stocks] = 1.0

        return X_aug_all, y120_all, ret120_all, mask_all

    def _train(
        self,
        model: CrossSectionalMLP,
        X_tr, y_tr, X_va, y_va,
        ts_rows_tr, ts_counts_tr, n_ts_tr,
        ts_rows_va, ts_counts_va, n_ts_va,
        cs_cfg, key: jax.Array,
    ) -> CrossSectionalMLP:
        smoothing = cs_cfg.label_smoothing
        ln_weight = float(cs_cfg.listnet_weight)
        ln_tau_true = float(cs_cfg.listnet_tau_true)
        ln_tau_pred = float(cs_cfg.listnet_tau_pred)
        h_weight_120 = float(cs_cfg.horizon_weight_120)
        n_ts_per_batch = int(cs_cfg.n_timestamps_per_batch)
        freeze_body = getattr(cs_cfg, 'freeze_body', False)

        warmup_epochs = getattr(cs_cfg, 'warmup_epochs', 0)
        if warmup_epochs > 0:
            steps_per_epoch = max(1, n_ts_tr // n_ts_per_batch)
            warmup_steps = warmup_epochs * steps_per_epoch
            total_steps = cs_cfg.epochs * steps_per_epoch
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=cs_cfg.lr * 0.01,
                peak_value=cs_cfg.lr,
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
                end_value=cs_cfg.lr * 0.01,
            )
            optimizer = optax.adamw(schedule, weight_decay=cs_cfg.weight_decay)
        else:
            optimizer = optax.adamw(cs_cfg.lr, weight_decay=cs_cfg.weight_decay)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def train_step(model, opt_state, x_ts, y120_ts, ret120_ts, mask_ts, key):
            """CE + ListNet training step on padded [N_ts, S, F*3] tensors."""
            def loss_fn(model):
                def forward_stock(xi, ki):
                    logits = model(xi, key=ki, inference=False)  # [H, C]
                    probs_120 = jax.nn.softmax(logits[1], axis=-1)
                    weights = jnp.arange(probs_120.shape[0], dtype=jnp.float32)
                    score = jnp.sum(probs_120 * weights)
                    return logits, score

                def forward_ts(x_stocks, keys_stocks):
                    return jax.vmap(forward_stock)(x_stocks, keys_stocks)

                # Forward all stocks across all timestamps
                keys_all = jax.random.split(key, x_ts.shape[0] * x_ts.shape[1])
                keys_all = keys_all.reshape(x_ts.shape[0], x_ts.shape[1], 2)
                all_logits, all_scores = jax.vmap(forward_ts)(x_ts, keys_all)
                # all_logits: [N_ts, S, H, C], all_scores: [N_ts, S]

                # CE loss (masked)
                logits_120 = all_logits[:, :, 1, :]  # [N_ts, S, C]
                ce_per_stock = optax.softmax_cross_entropy_with_integer_labels(
                    logits_120.reshape(-1, logits_120.shape[-1]),
                    y120_ts.reshape(-1),
                ).reshape(x_ts.shape[0], x_ts.shape[1])
                # Apply label smoothing manually if needed
                ce_masked = ce_per_stock * mask_ts
                ce_loss = jnp.sum(ce_masked) / jnp.maximum(jnp.sum(mask_ts), 1.0)

                if ln_weight > 0:
                    ln_loss = listnet_loss(
                        all_scores, ret120_ts, mask_ts,
                        tau_pred=ln_tau_pred, tau_true=ln_tau_true,
                    )
                    return ce_loss + ln_weight * ln_loss
                return ce_loss

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            if freeze_body:
                # Zero out body grads — keep only head gradients
                head_w, head_b = grads.head.weight, grads.head.bias
                grads = jax.tree.map(
                    lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x, grads
                )
                grads = eqx.tree_at(
                    lambda m: (m.head.weight, m.head.bias), grads, (head_w, head_b)
                )
            updates, new_opt_state = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            return eqx.apply_updates(model, updates), new_opt_state, loss

        @eqx.filter_jit
        def eval_step(model, x_ts, y120_ts, mask_ts, key):
            def forward_stock(xi, ki):
                logits = model(xi, key=ki, inference=True)
                return logits

            def forward_ts(x_stocks, keys_stocks):
                return jax.vmap(forward_stock)(x_stocks, keys_stocks)

            keys_all = jax.random.split(key, x_ts.shape[0] * x_ts.shape[1])
            keys_all = keys_all.reshape(x_ts.shape[0], x_ts.shape[1], 2)
            all_logits = jax.vmap(forward_ts)(x_ts, keys_all)
            # [N_ts, S, H, C]

            logits_120 = all_logits[:, :, 1, :]  # [N_ts, S, C]
            ce_per_stock = optax.softmax_cross_entropy_with_integer_labels(
                logits_120.reshape(-1, logits_120.shape[-1]),
                y120_ts.reshape(-1),
            ).reshape(x_ts.shape[0], x_ts.shape[1])

            preds = jnp.argmax(logits_120, axis=-1)  # [N_ts, S]
            correct = (preds == y120_ts).astype(jnp.float32)

            ce_masked = ce_per_stock * mask_ts
            correct_masked = correct * mask_ts
            n_valid = jnp.maximum(jnp.sum(mask_ts), 1.0)
            return jnp.sum(ce_masked) / n_valid, jnp.sum(correct_masked) / n_valid

        max_stocks = cs_cfg.max_stocks
        best_val_loss = float("inf")
        patience_counter = 0
        best_model = model

        for epoch in range(cs_cfg.epochs):
            rng = np.random.default_rng(self.cfg.seed + epoch)
            ts_perm = rng.permutation(n_ts_tr)

            epoch_loss = 0.0
            num_batches = 0

            for ts_start in range(0, n_ts_tr, n_ts_per_batch):
                ts_idx = ts_perm[ts_start:ts_start + n_ts_per_batch]
                X_aug, y120, ret120, mask = self._load_ts_batch(
                    X_tr, y_tr, ts_rows_tr, ts_counts_tr, ts_idx, max_stocks
                )
                if X_aug is None:
                    continue

                key, subkey = jax.random.split(key)
                model, opt_state, loss = train_step(
                    model, opt_state,
                    jnp.array(X_aug), jnp.array(y120),
                    jnp.array(ret120), jnp.array(mask),
                    subkey,
                )

                epoch_loss += float(loss)
                num_batches += 1

            avg_train_loss = epoch_loss / max(num_batches, 1)

            # Validation
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_count = 0
            for ts_start in range(0, n_ts_va, n_ts_per_batch * 2):
                ts_idx = np.arange(ts_start, min(ts_start + n_ts_per_batch * 2, n_ts_va))
                X_aug, y120, ret120, mask = self._load_ts_batch(
                    X_va, y_va, ts_rows_va, ts_counts_va, ts_idx, max_stocks
                )
                if X_aug is None:
                    continue

                key, val_key = jax.random.split(key)
                vl, va = eval_step(
                    model, jnp.array(X_aug), jnp.array(y120),
                    jnp.array(mask), val_key,
                )
                n_valid = float(np.sum(mask))
                val_loss_sum += float(vl) * n_valid
                val_acc_sum += float(va) * n_valid
                val_count += n_valid

            val_loss = val_loss_sum / max(val_count, 1)
            val_acc = val_acc_sum / max(val_count, 1)

            if epoch % 5 == 0 or epoch == cs_cfg.epochs - 1:
                log.info(
                    f"[CS-MLP] Epoch {epoch:3d} | train_loss={avg_train_loss:.6f} "
                    f"| val_loss={val_loss:.6f} | val_acc_120={val_acc:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cs_cfg.patience:
                    log.info(
                        f"[CS-MLP] Early stopping at epoch {epoch}. "
                        f"Best val_loss={best_val_loss:.6f} | val_acc_120={val_acc:.4f}"
                    )
                    break

        log.info(f"[CS-MLP] Training complete. Best val_loss={best_val_loss:.6f}")
        return best_model

    def _predict_timestamps(
        self, X, y, ts_rows, ts_counts, n_ts,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict all stocks across all timestamps with cross-sectional augmentation.

        Returns predictions ordered by original row indices for consistency.
        """
        assert self.model is not None
        model = self.model
        cs_cfg = self.cfg.cs_mlp
        n_ts_per_batch = int(cs_cfg.n_timestamps_per_batch) * 2

        # Collect (row_idx, probs, actual) pairs
        results: dict[int, tuple[np.ndarray, float]] = {}
        key = jax.random.PRNGKey(0)

        @eqx.filter_jit
        def predict_batch(model, x, key):
            def per_sample(xi, ki):
                logits = model(xi, key=ki, inference=True)
                return jax.nn.softmax(logits[1], axis=-1)  # [C]
            keys = jax.random.split(key, x.shape[0])
            return jax.vmap(per_sample)(x, keys)

        for ts_start in range(0, n_ts, n_ts_per_batch):
            ts_idx = np.arange(ts_start, min(ts_start + n_ts_per_batch, n_ts))

            for ts_i in ts_idx:
                n_stocks = int(ts_counts[ts_i])
                if n_stocks < 2:
                    continue
                rows = ts_rows[ts_i, :n_stocks]

                X_raw = np.array(X[rows]).astype(np.float32)
                if self._feature_indices is not None:
                    X_raw = X_raw[:, self._feature_indices]
                y_batch = np.array(y[rows])

                # Cross-sectional augmentation
                feature_mode = getattr(self, '_feature_mode', 'raw_mean_zscore')
                X_aug = self._augment_features(X_raw, feature_mode)
                X_aug = np.nan_to_num(X_aug, nan=0.0)

                key, subkey = jax.random.split(key)
                probs = np.array(predict_batch(model, jnp.array(X_aug), subkey))

                for j, row_idx in enumerate(rows):
                    results[int(row_idx)] = (probs[j], float(y_batch[j, self._h1]))

        # Sort by row index
        sorted_rows = sorted(results.keys())
        all_probs = np.stack([results[r][0] for r in sorted_rows])
        all_actuals = np.array([results[r][1] for r in sorted_rows])

        return all_probs, all_actuals

    def predict_npy(
        self, npy_dir: str | Path, split: str = "test"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with cross-sectional augmentation.

        Returns predictions for ALL rows in the split (including invalid ones),
        with uniform predictions for rows not covered by timestamp groups.
        """
        assert self.model is not None
        npy_dir = Path(npy_dir)
        cs_cfg = self.cfg.cs_mlp

        X = np.load(npy_dir / "X.npy", mmap_mode="r")
        y = np.load(npy_dir / "y.npy", mmap_mode="r")
        with open(npy_dir / "offsets.json") as f:
            offsets_dict = json.load(f)

        # Get all row indices for this split
        all_indices = []
        for stock_info in offsets_dict.values():
            if split == "train":
                s, e = stock_info["start"], stock_info["train_end"]
            elif split == "val":
                s, e = stock_info["train_end"], stock_info["val_end"]
            else:
                s, e = stock_info["val_end"], stock_info["end"]
            all_indices.extend(range(s, e))
        all_indices = sorted(all_indices)
        total_rows = len(all_indices)

        # Predict valid rows via timestamp groups
        ts_rows, ts_counts, n_ts = self._build_timestamp_groups(
            offsets_dict, y, split, cs_cfg.max_stocks
        )
        valid_probs, valid_actuals = self._predict_timestamps(
            X, y, ts_rows, ts_counts, n_ts
        )

        # Build row_idx → prediction mapping
        # _predict_timestamps returns results keyed by row_idx
        # Re-run to get the mapping
        pred_map: dict[int, np.ndarray] = {}
        actual_map: dict[int, float] = {}
        for i, row_idx in enumerate(sorted(set(
            int(ts_rows[t, s])
            for t in range(n_ts)
            for s in range(int(ts_counts[t]))
        ))):
            if i < len(valid_probs):
                pred_map[row_idx] = valid_probs[i]
                actual_map[row_idx] = valid_actuals[i]

        # Fill output arrays
        uniform = np.ones(self._num_classes, dtype=np.float32) / self._num_classes
        all_probs = np.zeros((total_rows, self._num_classes), dtype=np.float32)
        all_actuals = np.zeros(total_rows, dtype=np.float32)

        for i, row_idx in enumerate(all_indices):
            if row_idx in pred_map:
                all_probs[i] = pred_map[row_idx]
                all_actuals[i] = actual_map[row_idx]
            else:
                all_probs[i] = uniform
                all_actuals[i] = float(y[row_idx, self._h1])

        return all_probs, all_actuals

    def save(self, path: str | Path) -> None:
        assert self.model is not None
        assert self.feature_columns is not None

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        eqx.tree_serialise_leaves(str(path / "cs_mlp_model.eqx"), self.model)

        with open(path / "feature_columns.json", "w") as f:
            json.dump(self.feature_columns, f)

        cs_cfg = self.cfg.cs_mlp
        with open(path / "cs_mlp_config.json", "w") as f:
            json.dump({
                "in_features": len(self.feature_columns),
                "hidden_units": list(cs_cfg.hidden_units),
                "dropout": cs_cfg.dropout,
                "num_classes": self._num_classes,
                "num_horizons": self._num_horizons,
                "class_bins": list(self._class_interior_bins),
                "feature_mode": getattr(self, '_feature_mode', 'raw_mean_zscore'),
                "use_layernorm": getattr(self, '_use_layernorm', False),
                "activation": getattr(self, '_activation', 'gelu'),
                "feature_indices": self._feature_indices.tolist() if self._feature_indices is not None else None,
                "residual": getattr(self.model, 'residual', False),
            }, f)

        log.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)

        with open(path / "feature_columns.json") as f:
            self.feature_columns = json.load(f)
        with open(path / "cs_mlp_config.json") as f:
            cfg = json.load(f)

        key = jax.random.PRNGKey(0)
        feature_mode = cfg.get("feature_mode", "raw_mean_zscore")
        raw_features = cfg["in_features"]
        aug_dim = self._aug_dim(raw_features, feature_mode)
        self.model = CrossSectionalMLP(
            in_features=aug_dim,
            hidden_units=cfg["hidden_units"],
            dropout=cfg["dropout"],
            num_classes=cfg["num_classes"],
            num_horizons=cfg["num_horizons"],
            key=key,
            use_layernorm=cfg.get("use_layernorm", False),
            activation=cfg.get("activation", "gelu"),
            residual=cfg.get("residual", False),
        )
        self.model = eqx.tree_deserialise_leaves(
            str(path / "cs_mlp_model.eqx"), self.model
        )
        self._class_interior_bins = np.array(cfg["class_bins"], dtype=np.float64)
        self._feature_mode = cfg.get("feature_mode", "raw_mean_zscore")
        fi = cfg.get("feature_indices")
        self._feature_indices = np.array(fi, dtype=int) if fi is not None else None
        log.info(f"Model loaded from {path} (feature_mode={self._feature_mode})")
