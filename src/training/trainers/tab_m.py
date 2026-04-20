"""TabM trainer: PLE-B + BatchEnsemble/MiniEnsemble MLP (npy/memmap only).

Stage 1: Compute PLE boundaries from raw training data quantiles.
Stage 2: Train TabM or TabMmini.

Training: all K members forward per sample, loss on all K predictions.
Inference: softmax each member → average probabilities.
Gradient clipping: global norm = 1.0.
"""

import json
import logging
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from dataclasses import asdict

from src.config.models.tab_m import TabMConfig
from src.models.architectures.tab_m import TabM, TabMmini
from src.training.checkpoint import save_best_checkpoint
from src.training.registry import BaseTrainer, register_trainer
from src.data.labels import to_classes
from src.training.losses import (
    label_smoothing_ce, ordinal_emd_loss, listnet_loss, jax_rankdata,
)

log = logging.getLogger(__name__)


@register_trainer("tab_m", config_key="tab_m", model_type="cls", default_config=asdict(TabMConfig()))
class TabMTrainer(BaseTrainer):
    """TabM Trainer: PLE-B + BatchEnsemble/MiniEnsemble MLP."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        ds = cfg.dataset
        self._h0, self._h1, self._h2 = ds.horizon_indices
        self._vol_idx = ds.vol_idx
        self._num_horizons = len(ds.horizon_indices)
        self._class_interior_bins = np.array(ds.class_bins, dtype=np.float64)
        self._num_classes = len(ds.class_bins) + 1
        self.model: TabM | TabMmini | None = None
        self.feature_columns: list[str] | None = None
        self.ple_boundaries: np.ndarray | None = None
        self._feat_min: np.ndarray | None = None
        self._feat_max: np.ndarray | None = None
        self._val_predictions: np.ndarray | None = None
        self._val_actual: np.ndarray | None = None


    def fit_from_npy(self, npy_dir: str | Path) -> None:
        npy_dir = Path(npy_dir)
        tab_cfg = self.cfg.tab_m

        in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices = \
            self._load_data(npy_dir, tab_cfg)

        key = jax.random.PRNGKey(self.cfg.seed)
        key, model_key = jax.random.split(key)

        # Stage 1: PLE boundaries from raw training data
        n_bins = tab_cfg.n_bins
        log.info(f"Computing PLE boundaries (n_bins={n_bins})...")
        self.ple_boundaries = self._compute_ple_boundaries(X_train, tr_indices, n_bins)
        log.info(f"PLE boundaries: {self.ple_boundaries.shape}")

        # Stage 2: Build and train
        variant = getattr(tab_cfg, "variant", "tabm")
        model_cls = TabMmini if variant == "mini" else TabM
        d_embedding = getattr(tab_cfg, "d_embedding", 16)
        log.info(f"=== TabM Training (variant={variant}, K={tab_cfg.num_members}, "
                 f"d_block={tab_cfg.d_block}, n_blocks={tab_cfg.n_blocks}, "
                 f"d_embedding={d_embedding}, n_bins={n_bins}) ===")
        reg_weight = float(getattr(tab_cfg, "reg_weight", 0.0))
        use_reg = reg_weight > 0 and variant != "mini"
        model_kwargs = dict(
            num_features=in_features,
            num_members=tab_cfg.num_members,
            d_block=tab_cfg.d_block,
            d_embedding=d_embedding,
            n_blocks=tab_cfg.n_blocks,
            input_dropout=tab_cfg.input_dropout,
            block_dropout=tab_cfg.block_dropout,
            num_classes=self._num_classes, num_horizons=self._num_horizons,
            ple_boundaries=self.ple_boundaries,
            key=model_key,
        )
        if variant != "mini":
            model_kwargs["use_reg_head"] = use_reg
            model_kwargs["use_skip"] = bool(getattr(tab_cfg, "use_skip", False))
        self.model = model_cls(**model_kwargs)
        # Build timestamp groups for ListNet (if enabled)
        listnet_weight = float(getattr(tab_cfg, "listnet_weight", 0.0))
        ts_train_data = None
        ts_val_data = None
        if listnet_weight > 0:
            use_offsets = (Path(npy_dir) / "X.npy").exists() and (Path(npy_dir) / "offsets.json").exists()
            if use_offsets:
                with open(Path(npy_dir) / "offsets.json") as f:
                    offsets_dict = json.load(f)
                y_full = np.load(Path(npy_dir) / "y.npy", mmap_mode="r")
                ts_rows_tr, ts_counts_tr, n_ts_tr = self._build_timestamp_groups(offsets_dict, y_full, "train")
                ts_rows_va, ts_counts_va, n_ts_va = self._build_timestamp_groups(offsets_dict, y_full, "val")
                ts_train_data = (ts_rows_tr, ts_counts_tr, n_ts_tr)
                ts_val_data = (ts_rows_va, ts_counts_va, n_ts_va)
            else:
                log.warning("ListNet requires offsets.json (260306 format). Falling back to CE only.")
                listnet_weight = 0.0

        self.model = self._train(
            self.model,
            X_train, y_train, tr_indices,
            X_val, y_val, va_indices,
            tab_cfg, key,
            ts_train_data=ts_train_data,
            ts_val_data=ts_val_data,
        )

        log.info("Computing val predictions...")
        self._val_predictions, self._val_actual = self._predict_split(
            X_val, y_val, va_indices, tab_cfg.batch_size
        )

    def _load_data(self, npy_dir: Path, tab_cfg):
        in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices = \
            self._load_npy_data(npy_dir)

        # Load pre-computed feat_min/feat_max for PLE boundary endpoints
        scaler_path = npy_dir / "scaler_params.npz"
        if scaler_path.exists():
            scaler = np.load(scaler_path)
            if "feat_min" in scaler and "feat_max" in scaler:
                self._feat_min = scaler["feat_min"]
                self._feat_max = scaler["feat_max"]
                log.info("Loaded feat_min/feat_max from scaler_params.npz")

        return in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices

    def _build_timestamp_groups(self, offsets: dict, y: np.ndarray, split: str = "train"):
        """Build timestamp → row_indices mapping for cross-sectional training.

        All stocks' train_end aligns to the same calendar date, so
        relative_pos = max_train_len - stock_train_len + t maps to the same timestamp.

        Returns: (ts_rows, ts_counts, n_timestamps)
            ts_rows: [n_timestamps, max_stocks] padded row indices
            ts_counts: [n_timestamps] valid stock count per timestamp
        """
        tab_cfg = self.cfg.tab_m
        max_stocks = tab_cfg.max_stocks

        # Find max train length (reference for alignment)
        if split == "train":
            lengths = {k: v["train_end"] - v["start"] for k, v in offsets.items()}
        else:
            lengths = {k: v["val_end"] - v["train_end"] for k, v in offsets.items()}
        max_len = max(lengths.values())

        # Build mapping: relative_timestamp → list of absolute row indices
        ts_to_rows: dict[int, list[int]] = {}
        for stock_id, info in offsets.items():
            if split == "train":
                s_start, s_end = info["start"], info["train_end"]
            else:
                s_start, s_end = info["train_end"], info["val_end"]
            s_len = s_end - s_start
            if s_len == 0:
                continue
            offset = max_len - s_len  # align to end
            for t in range(s_len):
                abs_row = s_start + t
                # Filter invalid rows
                if y[abs_row, self._vol_idx] == 0 or y[abs_row, self._h1] == 0:
                    continue
                rel_t = offset + t
                if rel_t not in ts_to_rows:
                    ts_to_rows[rel_t] = []
                ts_to_rows[rel_t].append(abs_row)

        # Convert to padded arrays
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

    def _load_listnet_batch(self, X, y, ts_rows, ts_counts, ts_indices):
        """Load timestamp-grouped batch for ListNet training.

        ts_indices: indices into ts_rows (which timestamps to load)

        Returns: (X_batch, y60, y120, y240, returns_120, mask)
            X_batch: [N_ts, max_stocks, F]
            y60/y120/y240: [N_ts, max_stocks] class labels
            returns_120: [N_ts, max_stocks] raw returns
            mask: [N_ts, max_stocks] 1=valid, 0=padding
        """
        max_stocks = self.cfg.tab_m.max_stocks
        bins = self._class_interior_bins
        N = len(ts_indices)
        F = X.shape[1]

        X_batch = np.zeros((N, max_stocks, F), dtype=np.float32)
        y60 = np.zeros((N, max_stocks), dtype=np.int32)
        y120 = np.zeros((N, max_stocks), dtype=np.int32)
        y240 = np.zeros((N, max_stocks), dtype=np.int32)
        returns_120 = np.zeros((N, max_stocks), dtype=np.float32)
        mask = np.zeros((N, max_stocks), dtype=np.float32)

        for i, ts_idx in enumerate(ts_indices):
            n_valid = int(ts_counts[ts_idx])
            rows = ts_rows[ts_idx, :n_valid]
            X_b = np.array(X[rows]).astype(np.float32)
            X_b = np.nan_to_num(X_b, nan=0.0)
            y_b = np.array(y[rows])

            X_batch[i, :n_valid] = X_b
            y60[i, :n_valid] = to_classes(y_b[:, self._h0], bins)
            y120[i, :n_valid] = to_classes(y_b[:, self._h1], bins)
            y240[i, :n_valid] = to_classes(y_b[:, self._h2], bins)
            returns_120[i, :n_valid] = y_b[:, self._h1].astype(np.float32)
            mask[i, :n_valid] = 1.0

        return X_batch, y60, y120, y240, returns_120, mask

    def _compute_ple_boundaries(self, X, indices, n_bins, max_samples=2_000_000):
        rng = np.random.default_rng(self.cfg.seed + 999)
        if len(indices) > max_samples:
            sample_idx = rng.choice(indices, max_samples, replace=False)
        else:
            sample_idx = indices

        X_sample = np.array(X[sample_idx]).astype(np.float32)
        X_sample = np.nan_to_num(X_sample, nan=0.0)

        n_features = X_sample.shape[1]
        quantiles = np.linspace(0, 1, n_bins + 1)
        boundaries = np.zeros((n_features, n_bins + 1), dtype=np.float32)

        for j in range(n_features):
            boundaries[j] = np.quantile(X_sample[:, j], quantiles)
            # Ensure strictly increasing
            for k in range(1, n_bins + 1):
                if boundaries[j, k] <= boundaries[j, k - 1]:
                    boundaries[j, k] = boundaries[j, k - 1] + 1e-6

        # Use pre-computed full training set min/max for endpoints
        if self._feat_min is not None:
            boundaries[:, 0] = self._feat_min
        if self._feat_max is not None:
            boundaries[:, -1] = self._feat_max

        return boundaries

    def _train(self, model, X_tr, y_tr, tr_idx, X_va, y_va, va_idx, tab_cfg, key,
               ts_train_data=None, ts_val_data=None):
        weight_decay = tab_cfg.weight_decay
        lr_schedule = getattr(tab_cfg, "lr_schedule", "constant")
        steps_per_epoch = len(tr_idx) // tab_cfg.batch_size + 1
        total_steps = steps_per_epoch * tab_cfg.epochs

        if lr_schedule == "cosine":
            lr = optax.cosine_decay_schedule(init_value=tab_cfg.lr, decay_steps=total_steps, alpha=1e-6)
        else:
            lr = tab_cfg.lr

        # AdamW + gradient clipping (norm=1.0)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr, weight_decay=weight_decay),
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        batch_size = tab_cfg.batch_size
        smoothing = tab_cfg.label_smoothing
        horizon_weight_120 = float(getattr(tab_cfg, "horizon_weight_120", 1.0))
        cw_list = list(getattr(tab_cfg, "class_weights", []))
        class_weights = jnp.array(cw_list, dtype=jnp.float32) if cw_list else None
        focal_gamma = float(getattr(tab_cfg, "focal_gamma", 0.0))
        ordinal_weight = float(getattr(tab_cfg, "ordinal_weight", 0.0))
        if class_weights is not None:
            log.info(f"Class weights: {cw_list}")
        if focal_gamma > 0:
            log.info(f"Focal gamma: {focal_gamma}")
        reg_weight = float(getattr(tab_cfg, "reg_weight", 0.0))
        has_reg_head = reg_weight > 0 and getattr(tab_cfg, "variant", "tabm") != "mini"
        if ordinal_weight > 0:
            log.info(f"Ordinal EMD weight: {ordinal_weight}")
        if has_reg_head:
            log.info(f"Regression aux weight: {reg_weight} (MSE on c2c_120)")

        @eqx.filter_jit
        def train_step(model, opt_state, x, y60, y120, y240, r120, key):
            def loss_fn(model):
                def per_sample(xi, l60, l120, l240, raw120, ki):
                    if has_reg_head:
                        all_logits, all_reg = model(xi, key=ki, inference=False)
                    else:
                        all_logits = model(xi, key=ki, inference=False)

                    # Loss on each member independently
                    def member_loss(k_idx):
                        logits_k = all_logits[k_idx]
                        ce_60  = label_smoothing_ce(logits_k[0:1], l60[None], smoothing, class_weights, focal_gamma)
                        ce_120 = label_smoothing_ce(logits_k[1:2], l120[None], smoothing, class_weights, focal_gamma)
                        ce_240 = label_smoothing_ce(logits_k[2:3], l240[None], smoothing, class_weights, focal_gamma)
                        loss = (ce_60 + horizon_weight_120 * ce_120 + ce_240) / (2.0 + horizon_weight_120)
                        if ordinal_weight > 0:
                            emd_60  = ordinal_emd_loss(logits_k[0:1], l60[None])
                            emd_120 = ordinal_emd_loss(logits_k[1:2], l120[None])
                            emd_240 = ordinal_emd_loss(logits_k[2:3], l240[None])
                            emd = (emd_60 + horizon_weight_120 * emd_120 + emd_240) / (2.0 + horizon_weight_120)
                            loss = loss + ordinal_weight * emd
                        if has_reg_head:
                            reg_k = all_reg[k_idx]  # [H]
                            mse_120 = (reg_k[1] - raw120) ** 2
                            loss = loss + reg_weight * mse_120
                        return loss

                    K = all_logits.shape[0]
                    losses = jax.vmap(member_loss)(jnp.arange(K))  # [K]
                    return jnp.mean(losses)

                keys = jax.random.split(key, x.shape[0])
                losses = jax.vmap(per_sample)(x, y60, y120, y240, r120, keys)
                return jnp.mean(losses)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, new_opt_state = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            return eqx.apply_updates(model, updates), new_opt_state, loss

        @eqx.filter_jit
        def eval_step(model, x, y60, y120, y240, key):
            def per_sample(xi, l60, l120, l240, ki):
                # Inference: returns [H, C] averaged probabilities
                avg_probs = model(xi, key=ki, inference=True)
                ce_120 = optax.softmax_cross_entropy(avg_probs[1], jax.nn.one_hot(l120, avg_probs.shape[-1]))
                pred_120 = jnp.argmax(avg_probs[1])
                correct_120 = (pred_120 == l120).astype(jnp.float32)
                return ce_120, correct_120

            keys = jax.random.split(key, x.shape[0])
            losses, corrects = jax.vmap(per_sample)(x, y60, y120, y240, keys)
            return jnp.mean(losses), jnp.mean(corrects)

        # ── ListNet setup ──
        ln_weight = float(getattr(tab_cfg, "listnet_weight", 0.0))
        use_listnet = ln_weight > 0 and ts_train_data is not None
        if use_listnet:
            ln_tau_true = float(getattr(tab_cfg, "listnet_tau_true", 0.1))
            ln_tau_pred = float(getattr(tab_cfg, "listnet_tau_pred", 1.0))
            n_ts_per_batch = int(getattr(tab_cfg, "n_timestamps_per_batch", 5))
            max_stocks = int(getattr(tab_cfg, "max_stocks", 820))
            ts_rows_tr, ts_counts_tr, n_ts_tr = ts_train_data
            log.info(f"ListNet enabled: weight={ln_weight}, tau_true={ln_tau_true}, "
                     f"tau_pred={ln_tau_pred}, ts/batch={n_ts_per_batch}")

            @eqx.filter_jit
            def train_step_listnet(model, opt_state, x_ts, y60_ts, y120_ts, y240_ts,
                                   returns_120, mask_ts, key):
                """ListNet training step: CE + listnet over timestamp-grouped batch.

                x_ts: [N_ts, S, F]
                y120_ts: [N_ts, S]
                returns_120: [N_ts, S]
                mask_ts: [N_ts, S]
                """
                def loss_fn(model):
                    # Forward all stocks across all timestamps
                    def forward_stock(xi, ki):
                        # inference=False: dropout active, returns [K, H, C]
                        all_logits = model(xi, key=ki, inference=False)  # [K, H, C]
                        # Average member logits for CE, then softmax for ListNet
                        return all_logits  # [K, H, C]

                    def forward_timestamp(x_stocks, keys_stocks, m):
                        # Only process valid stocks (mask), but we pad so vmap works
                        return jax.vmap(forward_stock)(x_stocks, keys_stocks)  # [S, K, H, C]

                    n_ts = x_ts.shape[0]
                    s = x_ts.shape[1]
                    all_keys = jax.random.split(key, n_ts * s).reshape(n_ts, s, 2)
                    # [N_ts, S, K, H, C]
                    all_logits = jax.vmap(
                        lambda xs, ks: jax.vmap(forward_stock)(xs, ks)
                    )(x_ts, all_keys)

                    K = all_logits.shape[2]

                    # ── CE loss (per-sample, per-member, masked) ──
                    def ce_per_member(k_idx):
                        # all_logits[:, :, k_idx, 1, :] = [N_ts, S, C] for c2c_120
                        logits_120 = all_logits[:, :, k_idx, 1, :]  # [N_ts, S, C]
                        # Flatten for CE
                        flat_logits = logits_120.reshape(-1, logits_120.shape[-1])  # [N*S, C]
                        flat_labels = y120_ts.reshape(-1)  # [N*S]
                        flat_mask = mask_ts.reshape(-1)  # [N*S]
                        # Per-sample CE
                        n_classes = flat_logits.shape[-1]
                        smooth_val = smoothing / n_classes
                        one_hot = jax.nn.one_hot(flat_labels, n_classes)
                        soft_labels = one_hot * (1.0 - smoothing) + smooth_val
                        log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
                        per_sample_ce = -jnp.sum(soft_labels * log_probs, axis=-1)
                        # Masked mean
                        return jnp.sum(per_sample_ce * flat_mask) / jnp.maximum(jnp.sum(flat_mask), 1.0)

                    ce_losses = jax.vmap(ce_per_member)(jnp.arange(K))  # [K]
                    ce_loss = jnp.mean(ce_losses)

                    # ── ListNet loss (cross-sectional, on averaged probs) ──
                    # Average member probs for score computation
                    all_probs = jax.nn.softmax(all_logits, axis=-1)  # [N_ts, S, K, H, C]
                    avg_probs = jnp.mean(all_probs, axis=2)  # [N_ts, S, H, C]
                    # score4 = P[3]+P[4]-P[0]-P[1] on c2c_120 (horizon=1)
                    probs_120 = avg_probs[:, :, 1, :]  # [N_ts, S, C]
                    scores = probs_120[:, :, 3] + probs_120[:, :, 4] \
                           - probs_120[:, :, 0] - probs_120[:, :, 1]  # [N_ts, S]

                    ln_loss = listnet_loss(scores, returns_120, mask_ts,
                                           tau_pred=ln_tau_pred, tau_true=ln_tau_true)

                    total = ce_loss + ln_weight * ln_loss
                    return total, (ce_loss, ln_loss)

                (loss, (ce_val, ln_val)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
                updates, new_opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(model, eqx.is_array)
                )
                return eqx.apply_updates(model, updates), new_opt_state, loss, ce_val, ln_val

        best_val_loss = float("inf")
        patience_counter = 0
        best_model = model

        for epoch in range(tab_cfg.epochs):
            rng = np.random.default_rng(self.cfg.seed + 10000 + epoch)
            epoch_loss = 0.0
            num_batches = 0

            if use_listnet:
                # ── ListNet training loop: timestamp-grouped batches ──
                ts_perm = rng.permutation(n_ts_tr)
                for ts_start in range(0, n_ts_tr, n_ts_per_batch):
                    ts_idx = ts_perm[ts_start:ts_start + n_ts_per_batch]
                    if len(ts_idx) < 2:
                        continue  # need at least 2 timestamps
                    X_ts, y60_ts, y120_ts, y240_ts, ret120_ts, mask_ts = \
                        self._load_listnet_batch(X_tr, y_tr, ts_rows_tr, ts_counts_tr, ts_idx)
                    key, subkey = jax.random.split(key)
                    model, opt_state, loss, ce_v, ln_v = train_step_listnet(
                        model, opt_state,
                        jnp.array(X_ts), jnp.array(y60_ts), jnp.array(y120_ts),
                        jnp.array(y240_ts), jnp.array(ret120_ts), jnp.array(mask_ts),
                        subkey,
                    )
                    epoch_loss += float(loss)
                    num_batches += 1

                    # Log CE vs ListNet scale on first batch of first epoch
                    if epoch == 0 and num_batches == 1:
                        log.info(f"[ListNet] First batch: CE={float(ce_v):.4f}, "
                                 f"ListNet={float(ln_v):.4f}, "
                                 f"weighted_ListNet={ln_weight * float(ln_v):.4f}")
            else:
                # ── Original CE training loop ──
                perm = rng.permutation(len(tr_idx))
                shuffled = tr_idx[perm]
                for start in range(0, len(shuffled), batch_size):
                    idx = shuffled[start:start + batch_size]
                    batch = self._load_batch(X_tr, y_tr, idx, return_raw=has_reg_head)
                    if has_reg_head:
                        X_b, y60_b, y120_b, y240_b, r120_b = batch
                    else:
                        X_b, y60_b, y120_b, y240_b = batch
                        r120_b = np.zeros(len(idx), dtype=np.float32)
                    key, subkey = jax.random.split(key)
                    model, opt_state, loss = train_step(
                        model, opt_state,
                        jnp.array(X_b),
                        jnp.array(y60_b), jnp.array(y120_b), jnp.array(y240_b),
                        jnp.array(r120_b),
                        subkey,
                    )
                    epoch_loss += float(loss)
                    num_batches += 1

            avg_train_loss = epoch_loss / max(num_batches, 1)

            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_count = 0
            for start in range(0, len(va_idx), batch_size * 2):
                idx = va_idx[start:start + batch_size * 2]
                X_b, y60_b, y120_b, y240_b = self._load_batch(X_va, y_va, idx)
                key, val_key = jax.random.split(key)
                vl, va = eval_step(
                    model,
                    jnp.array(X_b),
                    jnp.array(y60_b), jnp.array(y120_b), jnp.array(y240_b),
                    val_key,
                )
                val_loss_sum += float(vl) * len(idx)
                val_acc_sum += float(va) * len(idx)
                val_count += len(idx)
            val_loss = val_loss_sum / max(val_count, 1)
            val_acc = val_acc_sum / max(val_count, 1)

            if epoch % 5 == 0 or epoch == tab_cfg.epochs - 1:
                log.info(
                    f"[TabM] Epoch {epoch:3d} | train_loss={avg_train_loss:.6f} "
                    f"| val_loss={val_loss:.6f} | val_acc_120={val_acc:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                save_best_checkpoint(
                    trainer=self,
                    save_path=self._checkpoint_save_path,
                    best_model=best_model,
                    epoch=epoch,
                    val_loss=float(best_val_loss),
                    extra_meta={"val_acc": float(val_acc)},
                    logger=log,
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= tab_cfg.patience:
                    log.info(
                        f"[TabM] Early stopping at epoch {epoch}. "
                        f"Best val_loss={best_val_loss:.6f} | val_acc_120={val_acc:.4f}"
                    )
                    break

        log.info(f"[TabM] Training complete. Best val_loss={best_val_loss:.6f}")
        return best_model

    def _load_batch(self, X, y, idx, return_raw=False):
        X_b = np.array(X[idx]).astype(np.float32)
        X_b = np.nan_to_num(X_b, nan=0.0)
        y_b = np.array(y[idx])
        bins = self._class_interior_bins
        result = (X_b, to_classes(y_b[:, self._h0], bins), to_classes(y_b[:, self._h1], bins), to_classes(y_b[:, self._h2], bins))
        if return_raw:
            return result + (y_b[:, self._h1].astype(np.float32),)
        return result

    def _predict_split(self, X, y, indices, batch_size):
        assert self.model is not None
        model = self.model
        all_probs = []
        all_actuals = []
        key = jax.random.PRNGKey(0)

        @eqx.filter_jit
        def predict_batch(model, x, key):
            def per_sample(xi, ki):
                # inference=True returns averaged probabilities [H, C]
                avg_probs = model(xi, key=ki, inference=True)
                return avg_probs[1]  # c2c_120 probs [C]
            keys = jax.random.split(key, x.shape[0])
            return jax.vmap(per_sample)(x, keys)

        for start in range(0, len(indices), batch_size * 2):
            idx = indices[start:start + batch_size * 2]
            X_b = np.array(X[idx]).astype(np.float32)
            X_b = np.nan_to_num(X_b, nan=0.0)
            key, subkey = jax.random.split(key)
            probs = np.array(predict_batch(model, jnp.array(X_b), subkey))
            all_probs.append(probs)
            all_actuals.append(np.array(y[idx, self._h1]))

        return np.concatenate(all_probs, axis=0), np.concatenate(all_actuals, axis=0)

    def predict_npy(self, npy_dir: str | Path, split: str = "test"):
        assert self.model is not None
        npy_dir = Path(npy_dir)
        tab_cfg = self.cfg.tab_m

        use_offsets = (npy_dir / "X.npy").exists() and (npy_dir / "offsets.json").exists()
        if use_offsets:
            X = np.load(npy_dir / "X.npy", mmap_mode="r")
            y = np.load(npy_dir / "y.npy", mmap_mode="r")
            with open(npy_dir / "offsets.json") as f:
                offsets = json.load(f)
            idx_list = []
            for stock_info in offsets.values():
                if split == "train":
                    idx_list.append(np.arange(stock_info["start"], stock_info["train_end"]))
                elif split == "val":
                    idx_list.append(np.arange(stock_info["train_end"], stock_info["val_end"]))
                else:
                    idx_list.append(np.arange(stock_info["val_end"], stock_info["end"]))
            all_indices = np.concatenate(idx_list)
        else:
            X = np.load(npy_dir / f"X_{split}.npy", mmap_mode="r")
            y = np.load(npy_dir / f"y_{split}.npy", mmap_mode="r")
            all_indices = np.arange(X.shape[0])

        return self._predict_split(X, y, all_indices, tab_cfg.batch_size)


    def save(self, path: str | Path) -> None:
        assert self.model is not None
        assert self.feature_columns is not None
        assert self.ple_boundaries is not None

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        eqx.tree_serialise_leaves(str(path / "tab_m_model.eqx"), self.model)

        with open(path / "feature_columns.json", "w") as f:
            json.dump(self.feature_columns, f)

        np.save(path / "ple_boundaries.npy", self.ple_boundaries)

        tab_cfg = self.cfg.tab_m
        with open(path / "tab_m_config.json", "w") as f:
            json.dump({
                "in_features": len(self.feature_columns),
                "variant": getattr(tab_cfg, "variant", "tabm"),
                "num_members": tab_cfg.num_members,
                "d_block": tab_cfg.d_block,
                "d_embedding": getattr(tab_cfg, "d_embedding", 16),
                "n_blocks": tab_cfg.n_blocks,
                "n_bins": tab_cfg.n_bins,
                "input_dropout": tab_cfg.input_dropout,
                "block_dropout": tab_cfg.block_dropout,
                "num_classes": self._num_classes,
                "num_horizons": self._num_horizons,
            }, f)
        log.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)

        with open(path / "feature_columns.json") as f:
            self.feature_columns = json.load(f)
        with open(path / "tab_m_config.json") as f:
            cfg = json.load(f)

        self.ple_boundaries = np.load(path / "ple_boundaries.npy")

        key = jax.random.PRNGKey(0)

        variant = cfg.get("variant", "tabm")
        model_cls = TabMmini if variant == "mini" else TabM
        self.model = model_cls(
            num_features=cfg["in_features"],
            num_members=cfg["num_members"],
            d_block=cfg["d_block"],
            d_embedding=cfg.get("d_embedding", 16),
            n_blocks=cfg["n_blocks"],
            input_dropout=cfg["input_dropout"],
            block_dropout=cfg["block_dropout"],
            num_classes=cfg["num_classes"], num_horizons=cfg["num_horizons"],
            ple_boundaries=self.ple_boundaries,
            key=key,
        )
        self.model = eqx.tree_deserialise_leaves(
            str(path / "tab_m_model.eqx"), self.model
        )
        log.info(f"Model loaded from {path}")
