"""FT-Transformer trainer: PLE-B tokenization + Transformer encoder (npy/memmap only).

Stage 1: Compute PLE boundaries from raw training data quantiles.
Stage 2: Train FT-Transformer with standard CE loss.

Single model (no ensemble). Lower LR (1e-4), cosine annealing.
Gradient clipping: global norm = 1.0.
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

from src.config.models.ft_transformer import FTTransformerConfig
from src.models.architectures.ft_transformer import FTTransformer
from src.training.registry import BaseTrainer, register_trainer
from src.data.labels import to_classes
from src.training.losses import label_smoothing_ce

log = logging.getLogger(__name__)


@register_trainer(
    "ft_transformer",
    config_key="ft_transformer",
    model_type="cls",
    default_config=asdict(FTTransformerConfig()),
)
class FTTransformerTrainer(BaseTrainer):
    """FT-Transformer Trainer: PLE-B tokenization + Transformer encoder."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        ds = cfg.dataset
        self._h0, self._h1, self._h2 = ds.horizon_indices
        self._vol_idx = ds.vol_idx
        self._num_horizons = len(ds.horizon_indices)
        self._class_interior_bins = np.array(ds.class_bins, dtype=np.float64)
        self._num_classes = len(ds.class_bins) + 1
        self.model: FTTransformer | None = None
        self.feature_columns: list[str] | None = None
        self.ple_boundaries: np.ndarray | None = None
        self._feat_min: np.ndarray | None = None
        self._feat_max: np.ndarray | None = None
        self._val_predictions: np.ndarray | None = None
        self._val_actual: np.ndarray | None = None

    def fit_from_npy(self, npy_dir: str | Path) -> None:
        npy_dir = Path(npy_dir)
        ft_cfg = self.cfg.ft_transformer

        in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices = \
            self._load_data(npy_dir)

        key = jax.random.PRNGKey(self.cfg.seed)
        key, model_key = jax.random.split(key)

        # Stage 1: PLE boundaries from raw training data
        n_bins = ft_cfg.n_bins
        log.info(f"Computing PLE boundaries (n_bins={n_bins})...")
        self.ple_boundaries = self._compute_ple_boundaries(X_train, tr_indices, n_bins)
        log.info(f"PLE boundaries: {self.ple_boundaries.shape}")

        # Stage 2: Build model
        log.info(
            f"=== FT-Transformer Training ("
            f"d_token={ft_cfg.d_token}, n_heads={ft_cfg.n_heads}, "
            f"n_layers={ft_cfg.n_layers}, d_embedding={ft_cfg.d_embedding}, "
            f"n_bins={n_bins}) ==="
        )
        self.model = FTTransformer(
            num_features=in_features,
            d_token=ft_cfg.d_token,
            n_heads=ft_cfg.n_heads,
            n_layers=ft_cfg.n_layers,
            d_ffn_multiplier=ft_cfg.d_ffn_multiplier,
            attention_dropout=ft_cfg.attention_dropout,
            ffn_dropout=ft_cfg.ffn_dropout,
            residual_dropout=ft_cfg.residual_dropout,
            d_embedding=ft_cfg.d_embedding,
            num_classes=self._num_classes,
            num_horizons=self._num_horizons,
            ple_boundaries=self.ple_boundaries,
            key=model_key,
        )

        import tempfile, time
        checkpoint_dir = Path(tempfile.gettempdir()) / f"ft_transformer_ckpt_{int(time.time())}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Best-model checkpoint dir: {checkpoint_dir}")

        self.model = self._train(
            self.model, X_train, y_train, tr_indices, X_val, y_val, va_indices, ft_cfg, key,
            checkpoint_dir=checkpoint_dir,
        )

        log.info("Computing val predictions...")
        self._val_predictions, self._val_actual = self._predict_split(
            X_val, y_val, va_indices, ft_cfg.batch_size
        )

    def _load_data(self, npy_dir: Path):
        in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices = \
            self._load_npy_data(npy_dir)

        scaler_path = npy_dir / "scaler_params.npz"
        if scaler_path.exists():
            scaler = np.load(scaler_path)
            if "feat_min" in scaler and "feat_max" in scaler:
                self._feat_min = scaler["feat_min"]
                self._feat_max = scaler["feat_max"]
                log.info("Loaded feat_min/feat_max from scaler_params.npz")

        return in_features, X_train, y_train, X_val, y_val, tr_indices, va_indices

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
            for k in range(1, n_bins + 1):
                if boundaries[j, k] <= boundaries[j, k - 1]:
                    boundaries[j, k] = boundaries[j, k - 1] + 1e-6

        if self._feat_min is not None:
            boundaries[:, 0] = self._feat_min
        if self._feat_max is not None:
            boundaries[:, -1] = self._feat_max

        return boundaries

    def _load_batch(self, X, y, idx):
        X_b = np.array(X[idx]).astype(np.float32)
        X_b = np.nan_to_num(X_b, nan=0.0)
        y_b = np.array(y[idx])
        bins = self._class_interior_bins
        return (
            X_b,
            to_classes(y_b[:, self._h0], bins),
            to_classes(y_b[:, self._h1], bins),
            to_classes(y_b[:, self._h2], bins),
        )

    def _train(self, model, X_tr, y_tr, tr_idx, X_va, y_va, va_idx, ft_cfg, key, checkpoint_dir: Path | None = None):
        steps_per_epoch = len(tr_idx) // ft_cfg.batch_size + 1
        total_steps = steps_per_epoch * ft_cfg.epochs

        lr_schedule = getattr(ft_cfg, "lr_schedule", "cosine")
        if lr_schedule == "cosine":
            lr = optax.cosine_decay_schedule(
                init_value=ft_cfg.lr, decay_steps=total_steps, alpha=1e-6
            )
        else:
            lr = ft_cfg.lr

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr, weight_decay=ft_cfg.weight_decay),
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        batch_size = ft_cfg.batch_size
        smoothing = ft_cfg.label_smoothing
        cw_list = list(getattr(ft_cfg, "class_weights", []))
        class_weights = jnp.array(cw_list, dtype=jnp.float32) if cw_list else None
        if class_weights is not None:
            log.info(f"Class weights: {cw_list}")

        @eqx.filter_jit
        def train_step(model, opt_state, x, y60, y120, y240, key):
            def loss_fn(model):
                def per_sample(xi, l60, l120, l240, ki):
                    # Single model: returns [H, C] logits
                    logits = model(xi, key=ki, inference=False)
                    ce_60 = label_smoothing_ce(logits[0:1], l60[None], smoothing, class_weights, 0.0)
                    ce_120 = label_smoothing_ce(logits[1:2], l120[None], smoothing, class_weights, 0.0)
                    ce_240 = label_smoothing_ce(logits[2:3], l240[None], smoothing, class_weights, 0.0)
                    return (ce_60 + ce_120 + ce_240) / 3.0

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
                # inference=True returns [H, C] probabilities
                probs = model(xi, key=ki, inference=True)
                ce_120 = optax.softmax_cross_entropy(
                    probs[1], jax.nn.one_hot(l120, probs.shape[-1])
                )
                pred_120 = jnp.argmax(probs[1])
                correct_120 = (pred_120 == l120).astype(jnp.float32)
                return ce_120, correct_120

            keys = jax.random.split(key, x.shape[0])
            losses, corrects = jax.vmap(per_sample)(x, y60, y120, y240, keys)
            return jnp.mean(losses), jnp.mean(corrects)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model = model

        for epoch in range(ft_cfg.epochs):
            rng = np.random.default_rng(self.cfg.seed + 10000 + epoch)
            perm = rng.permutation(len(tr_idx))
            shuffled = tr_idx[perm]
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, len(shuffled), batch_size):
                idx = shuffled[start:start + batch_size]
                X_b, y60_b, y120_b, y240_b = self._load_batch(X_tr, y_tr, idx)
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

            if epoch % 5 == 0 or epoch == ft_cfg.epochs - 1:
                log.info(
                    f"[FTTransformer] Epoch {epoch:3d} | train_loss={avg_train_loss:.6f} "
                    f"| val_loss={val_loss:.6f} | val_acc_120={val_acc:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                patience_counter = 0
                if checkpoint_dir is not None:
                    ckpt_path = checkpoint_dir / "best_model.eqx"
                    eqx.tree_serialise_leaves(str(ckpt_path), best_model)
                    log.info(f"[FTTransformer] Checkpoint saved (epoch={epoch}, val_loss={best_val_loss:.6f}) → {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= ft_cfg.patience:
                    log.info(
                        f"[FTTransformer] Early stopping at epoch {epoch}. "
                        f"Best val_loss={best_val_loss:.6f}"
                    )
                    break

        log.info(f"[FTTransformer] Training complete. Best val_loss={best_val_loss:.6f}")
        return best_model

    def _predict_split(self, X, y, indices, batch_size):
        assert self.model is not None
        model = self.model
        all_probs = []
        all_actuals = []
        key = jax.random.PRNGKey(0)

        @eqx.filter_jit
        def predict_batch(model, x, key):
            def per_sample(xi, ki):
                # inference=True returns [H, C] probabilities
                probs = model(xi, key=ki, inference=True)
                return probs[1]  # c2c_120 horizon [C]

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
        ft_cfg = self.cfg.ft_transformer

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

        return self._predict_split(X, y, all_indices, ft_cfg.batch_size)

    def save(self, path: str | Path) -> None:
        assert self.model is not None
        assert self.feature_columns is not None
        assert self.ple_boundaries is not None

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        eqx.tree_serialise_leaves(str(path / "ft_transformer_model.eqx"), self.model)

        with open(path / "feature_columns.json", "w") as f:
            json.dump(self.feature_columns, f)

        np.save(path / "ple_boundaries.npy", self.ple_boundaries)

        ft_cfg = self.cfg.ft_transformer
        with open(path / "ft_transformer_config.json", "w") as f:
            json.dump({
                "in_features": len(self.feature_columns),
                "d_token": ft_cfg.d_token,
                "n_heads": ft_cfg.n_heads,
                "n_layers": ft_cfg.n_layers,
                "d_ffn_multiplier": ft_cfg.d_ffn_multiplier,
                "attention_dropout": ft_cfg.attention_dropout,
                "ffn_dropout": ft_cfg.ffn_dropout,
                "residual_dropout": ft_cfg.residual_dropout,
                "d_embedding": ft_cfg.d_embedding,
                "n_bins": ft_cfg.n_bins,
                "num_classes": self._num_classes,
                "num_horizons": self._num_horizons,
            }, f)
        log.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)

        with open(path / "feature_columns.json") as f:
            self.feature_columns = json.load(f)
        with open(path / "ft_transformer_config.json") as f:
            cfg = json.load(f)

        self.ple_boundaries = np.load(path / "ple_boundaries.npy")

        key = jax.random.PRNGKey(0)
        self.model = FTTransformer(
            num_features=cfg["in_features"],
            d_token=cfg["d_token"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ffn_multiplier=cfg["d_ffn_multiplier"],
            attention_dropout=cfg["attention_dropout"],
            ffn_dropout=cfg["ffn_dropout"],
            residual_dropout=cfg["residual_dropout"],
            d_embedding=cfg["d_embedding"],
            num_classes=cfg["num_classes"],
            num_horizons=cfg["num_horizons"],
            ple_boundaries=self.ple_boundaries,
            key=key,
        )
        self.model = eqx.tree_deserialise_leaves(
            str(path / "ft_transformer_model.eqx"), self.model
        )
        log.info(f"Model loaded from {path}")
