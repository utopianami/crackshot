"""2-stage SAE+MLP trainer (npy/memmap only).

Stage 1: SupervisedAutoencoder — reconstruction + classification loss.
Stage 2: FrozenEncoderMLP — classification only, encoder weights frozen via
          jax.lax.stop_gradient inside the model's __call__.
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

from src.config.models.ae_mlp import AEMLPConfig
from src.data.labels import to_classes
from src.data.quantile_normalization import fit_quantile_params, transform_np
from src.data.scaler_io import save_scaler_params, load_scaler_params
from src.models.architectures.ae_mlp import FrozenEncoderMLP, SupervisedAutoencoder
from src.training.losses import label_smoothing_ce
from src.training.registry import BaseTrainer, register_trainer

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────

@register_trainer("ae_mlp", config_key="ae_mlp", model_type="cls", default_config=asdict(AEMLPConfig()))
class AEMLPTrainer(BaseTrainer):
    """2-stage Supervised Autoencoder + MLP Trainer (npy memmap mode)."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        ds = cfg.dataset
        self._h0, self._h1, self._h2 = ds.horizon_indices
        self._vol_idx = ds.vol_idx
        self._num_horizons = len(ds.horizon_indices)
        self._class_interior_bins = np.array(ds.class_bins, dtype=np.float64)
        self._num_classes = len(ds.class_bins) + 1
        self.ae_model: SupervisedAutoencoder | None = None
        self.mlp_model: FrozenEncoderMLP | None = None
        self.feature_columns: list[str] | None = None
        self.scaler_params: dict | None = None
        self._val_predictions: np.ndarray | None = None
        self._val_actual: np.ndarray | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------


    def fit_from_npy(self, npy_dir: str | Path) -> None:
        npy_dir = Path(npy_dir)
        ae_cfg = self.cfg.ae_mlp

        # ── 1. Load data ─────────────────────────────────────────────
        in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices = \
            self._load_data(npy_dir, ae_cfg)

        key = jax.random.PRNGKey(self.cfg.seed)
        key, ae_key, mlp_key = jax.random.split(key, 3)

        # ── Stage 1: Check for pre-trained SAE or train new one ──────
        sae_path = getattr(ae_cfg, "sae_checkpoint", "")
        if sae_path and Path(sae_path).exists():
            log.info(f"=== Loading pre-trained SAE from {sae_path} ===")
            self.ae_model = SupervisedAutoencoder(
                num_features=in_features,
                bottleneck_dim=ae_cfg.bottleneck_dim,
                ae_hidden=ae_cfg.ae_hidden,
                ae_dropout=ae_cfg.ae_dropout,
                decoder_dropout=ae_cfg.decoder_dropout,
                noise_sigma=ae_cfg.noise_sigma,
                num_classes=self._num_classes, num_horizons=self._num_horizons, key=ae_key,
            )
            self.ae_model = eqx.tree_deserialise_leaves(
                str(Path(sae_path) / "ae_model.eqx"), self.ae_model
            )
            log.info("SAE loaded (skipping Stage 1)")
        else:
            self.ae_model = SupervisedAutoencoder(
                num_features=in_features,
                bottleneck_dim=ae_cfg.bottleneck_dim,
                ae_hidden=ae_cfg.ae_hidden,
                ae_dropout=ae_cfg.ae_dropout,
                decoder_dropout=ae_cfg.decoder_dropout,
                noise_sigma=ae_cfg.noise_sigma,
                num_classes=self._num_classes, num_horizons=self._num_horizons, key=ae_key,
            )
            log.info("=== Stage 1: Autoencoder training ===")
            self.ae_model = self._train_ae(
                self.ae_model,
                X_train, y_train, tr_indices,
                X_val, y_val, va_indices,
                ae_cfg, key,
            )
        key, _ = jax.random.split(key)

        # ── Stage 2: FrozenEncoderMLP training ──────────────────────
        log.info("=== Stage 2: FrozenEncoderMLP training ===")
        self.mlp_model = FrozenEncoderMLP(
            num_features=in_features,
            bottleneck_dim=ae_cfg.bottleneck_dim,
            mlp_hidden_units=list(ae_cfg.mlp_hidden_units),
            mlp_input_dropout=ae_cfg.mlp_input_dropout,
            mlp_dropout=ae_cfg.mlp_dropout,
            num_classes=self._num_classes, num_horizons=self._num_horizons,
            sae=self.ae_model, key=mlp_key,
        )
        self.mlp_model = self._train_mlp(
            self.mlp_model,
            X_train, y_train, tr_indices,
            X_val, y_val, va_indices,
            ae_cfg, key,
        )

        # ── Val predictions for evaluation ───────────────────────────
        log.info("Computing val predictions …")
        self._val_predictions, self._val_actual = self._predict_split(
            X_val, y_val, va_indices, ae_cfg.batch_size
        )

    def _load_data(self, npy_dir: Path, ae_cfg):
        in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices = \
            self._load_npy_data(npy_dir)

        # Fit quantile scaler (needs training indices)
        feat_min, feat_max = None, None
        scaler_path = npy_dir / "scaler_params.npz"
        if scaler_path.exists():
            scaler = np.load(scaler_path)
            if "feat_min" in scaler and "feat_max" in scaler:
                feat_min = scaler["feat_min"]
                feat_max = scaler["feat_max"]
                log.info("Loaded feat_min/feat_max from scaler_params.npz")

        log.info("Fitting quantile scaler on training data …")
        self.scaler_params = fit_quantile_params(
            X_train, tr_indices, n_quantiles=1000,
            output_distribution="uniform", seed=self.cfg.seed,
            feat_min=feat_min, feat_max=feat_max,
        )
        log.info("Quantile scaler fitted (n_quantiles=1000, output=uniform)")

        return in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def _train_ae(
        self,
        model: SupervisedAutoencoder,
        X_tr, y_tr, tr_idx,
        X_va, y_va, va_idx,
        ae_cfg, key: jax.Array,
    ) -> SupervisedAutoencoder:
        optimizer = optax.adam(ae_cfg.ae_lr)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        batch_size = ae_cfg.batch_size

        @eqx.filter_jit
        def train_step(model, opt_state, x, y60, y120, y240, key):
            def loss_fn(model):
                def per_sample(xi, l60, l120, l240, ki):
                    decoder_out, ae_pred, _ = model(xi, key=ki, inference=False)
                    recon_loss = jnp.mean((decoder_out - xi) ** 2)
                    ce_60  = optax.softmax_cross_entropy_with_integer_labels(ae_pred[0], l60)
                    ce_120 = optax.softmax_cross_entropy_with_integer_labels(ae_pred[1], l120)
                    ce_240 = optax.softmax_cross_entropy_with_integer_labels(ae_pred[2], l240)
                    cls_loss = (ce_60 + ce_120 + ce_240) / 3.0
                    return recon_loss + cls_loss

                keys = jax.random.split(key, x.shape[0])
                losses = jax.vmap(per_sample)(x, y60, y120, y240, keys)
                return jnp.mean(losses)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, new_opt_state = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            return eqx.apply_updates(model, updates), new_opt_state, loss

        @eqx.filter_jit
        def eval_step(model, x, y60, y120, y240, key):
            def per_sample(xi, l60, l120, l240, ki):
                decoder_out, ae_pred, _ = model(xi, key=ki, inference=True)
                recon_loss = jnp.mean((decoder_out - xi) ** 2)
                ce_60  = optax.softmax_cross_entropy_with_integer_labels(ae_pred[0], l60)
                ce_120 = optax.softmax_cross_entropy_with_integer_labels(ae_pred[1], l120)
                ce_240 = optax.softmax_cross_entropy_with_integer_labels(ae_pred[2], l240)
                cls_loss = (ce_60 + ce_120 + ce_240) / 3.0
                return recon_loss + cls_loss

            keys = jax.random.split(key, x.shape[0])
            losses = jax.vmap(per_sample)(x, y60, y120, y240, keys)
            return jnp.mean(losses)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model = model

        for epoch in range(ae_cfg.ae_epochs):
            rng = np.random.default_rng(self.cfg.seed + epoch)
            perm = rng.permutation(len(tr_idx))
            shuffled = tr_idx[perm]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, len(shuffled), batch_size):
                idx = shuffled[start:start + batch_size]
                X_b, y60_b, y120_b, y240_b, *_ = self._load_batch(X_tr, y_tr, idx)
                key, subkey = jax.random.split(key)
                model, opt_state, loss = train_step(
                    model, opt_state,
                    jnp.array(X_b),
                    jnp.array(y60_b), jnp.array(y120_b), jnp.array(y240_b),
                    subkey,
                )
                epoch_loss += float(loss)
                num_batches += 1

            avg_train_loss = epoch_loss / max(num_batches, 1)

            # Validation
            val_loss_sum = 0.0
            val_count = 0
            for start in range(0, len(va_idx), batch_size * 2):
                idx = va_idx[start:start + batch_size * 2]
                X_b, y60_b, y120_b, y240_b, *_ = self._load_batch(X_va, y_va, idx)
                key, val_key = jax.random.split(key)
                vl = float(eval_step(
                    model,
                    jnp.array(X_b),
                    jnp.array(y60_b), jnp.array(y120_b), jnp.array(y240_b),
                    val_key,
                ))
                val_loss_sum += vl * len(idx)
                val_count += len(idx)
            val_loss = val_loss_sum / max(val_count, 1)

            if epoch % 10 == 0 or epoch == ae_cfg.ae_epochs - 1:
                log.info(
                    f"[AE] Epoch {epoch:3d} | train_loss={avg_train_loss:.6f} | val_loss={val_loss:.6f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= ae_cfg.ae_patience:
                    log.info(f"[AE] Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.6f}")
                    break

        log.info(f"[AE] Training complete. Best val_loss={best_val_loss:.6f}")
        return best_model

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def _train_mlp(
        self,
        model: FrozenEncoderMLP,
        X_tr, y_tr, tr_idx,
        X_va, y_va, va_idx,
        ae_cfg, key: jax.Array,
    ) -> FrozenEncoderMLP:
        weight_decay = getattr(ae_cfg, "mlp_weight_decay", 0.0)
        lr_schedule = getattr(ae_cfg, "mlp_lr_schedule", "constant")
        steps_per_epoch = len(tr_idx) // ae_cfg.batch_size + 1
        total_steps = steps_per_epoch * ae_cfg.mlp_epochs

        if lr_schedule == "cosine":
            lr = optax.cosine_decay_schedule(
                init_value=ae_cfg.mlp_lr,
                decay_steps=total_steps,
                alpha=1e-6,  # minimum lr ratio
            )
        else:
            lr = ae_cfg.mlp_lr

        if weight_decay > 0:
            optimizer = optax.adamw(lr, weight_decay=weight_decay)
        else:
            optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        batch_size = ae_cfg.batch_size
        smoothing = ae_cfg.label_smoothing
        horizon_weight_120 = float(getattr(ae_cfg, "horizon_weight_120", 1.0))
        reg_weight = float(getattr(ae_cfg, "reg_weight", 0.0))
        cw_list = getattr(ae_cfg, "class_weights", [])
        cw = jnp.array(cw_list, dtype=jnp.float32) if cw_list else None
        focal_gamma = float(getattr(ae_cfg, "focal_gamma", 0.0))

        @eqx.filter_jit
        def train_step(model, opt_state, x, y60, y120, y240, r60, r120, r240, key):
            def loss_fn(model):
                def per_sample(xi, l60, l120, l240, raw60, raw120, raw240, ki):
                    cls_logits, reg_preds = model(xi, key=ki, inference=False)
                    ce_60  = label_smoothing_ce(cls_logits[0:1], l60[None], smoothing, cw, focal_gamma)
                    ce_120 = label_smoothing_ce(cls_logits[1:2], l120[None], smoothing, cw, focal_gamma)
                    ce_240 = label_smoothing_ce(cls_logits[2:3], l240[None], smoothing, cw, focal_gamma)
                    cls_loss = (ce_60 + horizon_weight_120 * ce_120 + ce_240) / (2.0 + horizon_weight_120)
                    if reg_weight > 0:
                        mse_60  = (reg_preds[0] - raw60) ** 2
                        mse_120 = (reg_preds[1] - raw120) ** 2
                        mse_240 = (reg_preds[2] - raw240) ** 2
                        reg_loss = (mse_60 + mse_120 + mse_240) / 3.0
                        return cls_loss + reg_weight * reg_loss
                    return cls_loss

                keys = jax.random.split(key, x.shape[0])
                losses = jax.vmap(per_sample)(x, y60, y120, y240, r60, r120, r240, keys)
                return jnp.mean(losses)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, new_opt_state = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            return eqx.apply_updates(model, updates), new_opt_state, loss

        @eqx.filter_jit
        def eval_step(model, x, y60, y120, y240, r60, r120, r240, key):
            def per_sample(xi, l60, l120, l240, raw60, raw120, raw240, ki):
                cls_logits, reg_preds = model(xi, key=ki, inference=True)
                ce_60  = optax.softmax_cross_entropy_with_integer_labels(cls_logits[0], l60)
                ce_120 = optax.softmax_cross_entropy_with_integer_labels(cls_logits[1], l120)
                ce_240 = optax.softmax_cross_entropy_with_integer_labels(cls_logits[2], l240)
                pred_120 = jnp.argmax(cls_logits[1])
                correct_120 = (pred_120 == l120).astype(jnp.float32)
                cls_loss = (ce_60 + ce_120 + ce_240) / 3.0
                if reg_weight > 0:
                    mse_60  = (reg_preds[0] - raw60) ** 2
                    mse_120 = (reg_preds[1] - raw120) ** 2
                    mse_240 = (reg_preds[2] - raw240) ** 2
                    reg_loss = (mse_60 + mse_120 + mse_240) / 3.0
                    return cls_loss + reg_weight * reg_loss, correct_120
                return cls_loss, correct_120

            keys = jax.random.split(key, x.shape[0])
            losses, corrects = jax.vmap(per_sample)(x, y60, y120, y240, r60, r120, r240, keys)
            return jnp.mean(losses), jnp.mean(corrects)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model = model

        for epoch in range(ae_cfg.mlp_epochs):
            rng = np.random.default_rng(self.cfg.seed + 10000 + epoch)
            perm = rng.permutation(len(tr_idx))
            shuffled = tr_idx[perm]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, len(shuffled), batch_size):
                idx = shuffled[start:start + batch_size]
                X_b, y60_b, y120_b, y240_b, r60_b, r120_b, r240_b = self._load_batch(X_tr, y_tr, idx)
                key, subkey = jax.random.split(key)
                model, opt_state, loss = train_step(
                    model, opt_state,
                    jnp.array(X_b),
                    jnp.array(y60_b), jnp.array(y120_b), jnp.array(y240_b),
                    jnp.array(r60_b), jnp.array(r120_b), jnp.array(r240_b),
                    subkey,
                )
                epoch_loss += float(loss)
                num_batches += 1

            avg_train_loss = epoch_loss / max(num_batches, 1)

            # Validation — monitor val_loss (more stable than accuracy)
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_count = 0
            for start in range(0, len(va_idx), batch_size * 2):
                idx = va_idx[start:start + batch_size * 2]
                X_b, y60_b, y120_b, y240_b, r60_b, r120_b, r240_b = self._load_batch(X_va, y_va, idx)
                key, val_key = jax.random.split(key)
                vl, va = eval_step(
                    model,
                    jnp.array(X_b),
                    jnp.array(y60_b), jnp.array(y120_b), jnp.array(y240_b),
                    jnp.array(r60_b), jnp.array(r120_b), jnp.array(r240_b),
                    val_key,
                )
                val_loss_sum += float(vl) * len(idx)
                val_acc_sum += float(va) * len(idx)
                val_count += len(idx)
            val_loss = val_loss_sum / max(val_count, 1)
            val_acc = val_acc_sum / max(val_count, 1)

            if epoch % 5 == 0 or epoch == ae_cfg.mlp_epochs - 1:
                log.info(
                    f"[MLP] Epoch {epoch:3d} | train_loss={avg_train_loss:.6f} "
                    f"| val_loss={val_loss:.6f} | val_acc_120={val_acc:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= ae_cfg.mlp_patience:
                    log.info(
                        f"[MLP] Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.6f} | val_acc_120={val_acc:.4f}"
                    )
                    break

        log.info(f"[MLP] Training complete. Best val_loss={best_val_loss:.6f}")
        return best_model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load, scale, and convert labels for a batch of indices.

        Returns: (X_b, cls_60, cls_120, cls_240, raw_60, raw_120, raw_240)
        """
        X_b = transform_np(np.array(X[idx]), self.scaler_params).astype(np.float32)
        X_b = np.nan_to_num(X_b, nan=0.0)
        y_b = np.array(y[idx])
        bins = self._class_interior_bins
        y60  = to_classes(y_b[:, self._h0], bins)
        y120 = to_classes(y_b[:, self._h1], bins)
        y240 = to_classes(y_b[:, self._h2], bins)
        raw60  = y_b[:, self._h0].astype(np.float32)
        raw120 = y_b[:, self._h1].astype(np.float32)
        raw240 = y_b[:, self._h2].astype(np.float32)
        return X_b, y60, y120, y240, raw60, raw120, raw240

    def _predict_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (probs_120 [N,5], raw_c2c_120 [N])."""
        assert self.mlp_model is not None
        model = self.mlp_model
        all_probs = []
        all_actuals = []
        key = jax.random.PRNGKey(0)

        @eqx.filter_jit
        def predict_batch(model, x, key):
            def per_sample(xi, ki):
                cls_logits, _reg_preds = model(xi, key=ki, inference=True)
                return jax.nn.softmax(cls_logits[1], axis=-1)   # c2c_120 probs [5]
            keys = jax.random.split(key, x.shape[0])
            return jax.vmap(per_sample)(x, keys)

        for start in range(0, len(indices), batch_size * 2):
            idx = indices[start:start + batch_size * 2]
            X_b = transform_np(np.array(X[idx]), self.scaler_params).astype(np.float32)
            X_b = np.nan_to_num(X_b, nan=0.0)
            key, subkey = jax.random.split(key)
            probs = np.array(predict_batch(model, jnp.array(X_b), subkey))
            all_probs.append(probs)
            all_actuals.append(np.array(y[idx, self._h1]))

        return np.concatenate(all_probs, axis=0), np.concatenate(all_actuals, axis=0)

    # ------------------------------------------------------------------
    # predict_npy
    # ------------------------------------------------------------------

    def predict_npy(
        self, npy_dir: str | Path, split: str = "test"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (probs [N,5], raw_c2c_120 [N])."""
        assert self.mlp_model is not None
        assert self.scaler_params is not None

        npy_dir = Path(npy_dir)
        ae_cfg = self.cfg.ae_mlp

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

        return self._predict_split(X, y, all_indices, ae_cfg.batch_size)

    # ------------------------------------------------------------------
    # predict (DataFrame) — not supported
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        assert self.ae_model is not None
        assert self.mlp_model is not None
        assert self.feature_columns is not None
        assert self.scaler_params is not None

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        eqx.tree_serialise_leaves(str(path / "ae_model.eqx"), self.ae_model)
        eqx.tree_serialise_leaves(str(path / "mlp_model.eqx"), self.mlp_model)

        with open(path / "feature_columns.json", "w") as f:
            json.dump(self.feature_columns, f)

        ae_cfg = self.cfg.ae_mlp
        with open(path / "ae_mlp_config.json", "w") as f:
            json.dump(
                {
                    "in_features": len(self.feature_columns),
                    "bottleneck_dim": ae_cfg.bottleneck_dim,
                    "ae_hidden": ae_cfg.ae_hidden,
                    "ae_dropout": ae_cfg.ae_dropout,
                    "decoder_dropout": ae_cfg.decoder_dropout,
                    "noise_sigma": ae_cfg.noise_sigma,
                    "mlp_hidden_units": list(ae_cfg.mlp_hidden_units),
                    "mlp_input_dropout": ae_cfg.mlp_input_dropout,
                    "mlp_dropout": ae_cfg.mlp_dropout,
                    "num_classes": self._num_classes,
                    "num_horizons": self._num_horizons,
                    "class_bins": list(self._class_interior_bins),
                },
                f,
            )

        save_scaler_params(path / "scaler_params.npz", self.scaler_params)
        log.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)

        with open(path / "feature_columns.json") as f:
            self.feature_columns = json.load(f)
        with open(path / "ae_mlp_config.json") as f:
            cfg = json.load(f)

        key = jax.random.PRNGKey(0)
        key, ae_key, mlp_key = jax.random.split(key, 3)

        self.ae_model = SupervisedAutoencoder(
            num_features=cfg["in_features"],
            bottleneck_dim=cfg["bottleneck_dim"],
            ae_hidden=cfg["ae_hidden"],
            ae_dropout=cfg["ae_dropout"],
            decoder_dropout=cfg["decoder_dropout"],
            noise_sigma=cfg["noise_sigma"],
            num_classes=cfg["num_classes"],
            num_horizons=cfg["num_horizons"],
            key=ae_key,
        )
        self.ae_model = eqx.tree_deserialise_leaves(
            str(path / "ae_model.eqx"), self.ae_model
        )

        self.mlp_model = FrozenEncoderMLP(
            num_features=cfg["in_features"],
            bottleneck_dim=cfg["bottleneck_dim"],
            mlp_hidden_units=cfg["mlp_hidden_units"],
            mlp_input_dropout=cfg["mlp_input_dropout"],
            mlp_dropout=cfg["mlp_dropout"],
            num_classes=cfg["num_classes"],
            num_horizons=cfg["num_horizons"],
            sae=self.ae_model,
            key=mlp_key,
        )
        self.mlp_model = eqx.tree_deserialise_leaves(
            str(path / "mlp_model.eqx"), self.mlp_model
        )

        # Restore class bins (interior format)
        class_bins = cfg.get("class_bins", [-0.01, -0.003, 0.003, 0.01])
        self._class_interior_bins = np.array(class_bins, dtype=np.float64)

        self.scaler_params = load_scaler_params(path / "scaler_params.npz")
        log.info(f"Model loaded from {path}")
