"""Training script with OmegaConf configuration.

Usage:
    # Config file + overrides
    PYTHONPATH="." uv run python scripts/train.py -c configs/tab_m.yaml tab_m.npy_dir=/path

    # Overrides only (uses dataclass defaults)
    PYTHONPATH="." uv run python scripts/train.py model.name=tab_m tab_m.npy_dir=/path tab_m.d_block=256

Determinism:
    GPU parallel reduction is non-deterministic by default in XLA/JAX, which
    makes same-seed retrains diverge (observed: ~3M alpha PnL spread across 3
    runs with identical command, 2026-04-08). To make training bit-exact
    reproducible, we append `--xla_gpu_deterministic_ops=true` to XLA_FLAGS
    before importing JAX. Set env var `DISABLE_DETERMINISM=1` to override
    (e.g. for speed benchmarking where reproducibility isn't needed).
"""

import os

if os.environ.get("DISABLE_DETERMINISM") != "1":
    _existing = os.environ.get("XLA_FLAGS", "")
    _det_flag = "--xla_gpu_deterministic_ops=true"
    if _det_flag not in _existing:
        os.environ["XLA_FLAGS"] = f"{_existing} {_det_flag}".strip()

import argparse
import logging
import time

import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

from src.config import load_config
from src.evaluation.metrics import calculate_metrics
from src.training.registry import (
    create_trainer,
    discover_trainers,
    is_classification,
)

log = logging.getLogger(__name__)


def _log_eval_cls(probs, actual, prefix=""):
    """Log evaluation for classification models (probs [N,5], actual [N] raw returns)."""
    try:
        from scripts.evaluate_topk import compute_score, topk_stats
    except ImportError:
        log.warning(f"{prefix}evaluate_topk not found, skipping topk eval")
        return
    score = compute_score(probs)
    # Filter vol==0 and c2c_120==0
    mask = actual != 0
    score_f, actual_f = score[mask], actual[mask]
    log.info(f"=== {prefix}Classification Eval (N={len(actual_f):,}, filtered {(~mask).sum():,} zero-label rows) ===")
    for k in [0.1, 0.5, 1, 5]:
        stats = topk_stats(score_f, actual_f, k)
        log.info(
            f"  k={k:>4.1f}% | Top dir={stats['top_dir_acc']:.1f}% ret={stats['top_mean_ret']:+.2f}% "
            f"| Bot dir={stats['bot_dir_acc']:.1f}% ret={stats['bot_mean_ret']:+.2f}%"
        )


def _log_eval(predictions, actual, prefix=""):
    n_total = len(predictions)
    n_up = int(np.sum(predictions > 0))
    n_down = n_total - n_up
    pcts = np.percentile(predictions, [1, 5, 25, 50, 75, 95, 99])
    log.info(f"=== {prefix}Output Distribution ===")
    log.info(f"  N={n_total}  UP={n_up} ({n_up/n_total*100:.1f}%)  DOWN={n_down} ({n_down/n_total*100:.1f}%)")
    log.info(f"  mean={predictions.mean():.6f}  std={predictions.std():.6f}")
    log.info(f"  min={predictions.min():.6f}  max={predictions.max():.6f}")
    log.info(f"  p1={pcts[0]:.6f}  p5={pcts[1]:.6f}  p25={pcts[2]:.6f}  p50={pcts[3]:.6f}  p75={pcts[4]:.6f}  p95={pcts[5]:.6f}  p99={pcts[6]:.6f}")

    metrics = calculate_metrics(predictions, actual)
    log.info(f"=== {prefix}Prediction Quality ===")
    log.info(f"  RMSE={metrics['rmse']:.6f}  MAE={metrics['mae']:.6f}")
    log.info(f"  Pearson={metrics['pearson_corr']:.4f}  Spearman={metrics['spearman_corr']:.4f}")
    log.info(f"  Direction: {metrics['direction_accuracy']:.1%} "
             f"(up: {metrics['precision_up']:.1%} [{metrics['n_pred_up']}], "
             f"down: {metrics['precision_down']:.1%} [{metrics['n_pred_down']}])")
    log.info("  --- Top k% (strongest BUY signals) ---")
    for k in ["0.1", "0.5", "1", "5", "10"]:
        log.info(f"  Top {k:>4s}%: acc={metrics[f'top{k}pct_acc']:.1%}, mean_ret={metrics[f'top{k}pct_mean_return']:.6f}")
    log.info("  --- Bot k% (strongest SELL signals) ---")
    for k in ["0.1", "0.5", "1", "5", "10"]:
        log.info(f"  Bot {k:>4s}%: acc={metrics[f'bot{k}pct_acc']:.1%}, mean_ret={metrics[f'bot{k}pct_mean_return']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("-c", "--config", type=str, default=None, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value dotlist)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Auto-discover trainers and load config
    discover_trainers()
    cfg = load_config(args.config, args.overrides)

    model_name = cfg.model.name
    log.info(f"Training model: {model_name}")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Resolve npy_dir from data config
    npy_dir = cfg.data.data_path
    if not npy_dir:
        raise ValueError("data.data_path must be set. Run convert_kospi_npy.py first.")

    # Timestamp suffix for unique output paths (used for both model save and predictions)
    from datetime import datetime
    ts_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Train
    trainer = create_trainer(model_name, cfg=cfg)

    # Inject save path so the trainer can overwrite-save "best" on every val
    # improvement (crash-safe). If training finishes cleanly, trainer.save()
    # is called again at the end below, which is a no-op rewrite of the same
    # best weights.
    save_path = Path(cfg.paths.saved_model_path) / f"{model_name}_{ts_suffix}"
    save_path.mkdir(parents=True, exist_ok=True)
    trainer._checkpoint_save_path = save_path
    log.info(f"Best-model checkpoint path: {save_path}")

    t_start = time.time()
    trainer.fit_from_npy(npy_dir)
    train_elapsed = time.time() - t_start
    log.info(f"Training time: {train_elapsed:.1f}s ({train_elapsed/60:.1f}min)")

    # 2. Evaluate (val)
    predictions = trainer._val_predictions
    actual = trainer._val_actual
    if is_classification(model_name):
        _log_eval_cls(predictions, actual, prefix="VAL ")
    else:
        _log_eval(predictions, actual, prefix="VAL ")

    # 3. Evaluate (test)
    npy_path = Path(npy_dir)
    has_test = (npy_path / "X_test.npy").exists() or (npy_path / "X.npy").exists()
    if has_test:
        test_predictions, test_actual = trainer.predict_npy(npy_dir, split="test")
        if is_classification(model_name):
            _log_eval_cls(test_predictions, test_actual, prefix="TEST ")
            # Save predictions for evaluate_topk.py
            out_dir = Path(cfg.paths.inference_path) / f"{model_name}_{ts_suffix}"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "test_probs.npy", test_predictions)
            np.save(out_dir / "test_actual.npy", test_actual)
            log.info(f"Test predictions saved to {out_dir}")
        else:
            _log_eval(test_predictions, test_actual, prefix="TEST ")

    # 4. Save final model to the same path used for best-model checkpointing.
    # The trainer's internal best_model has already been written to save_path
    # each time val improved, so this final save is an overwrite with the
    # best weights (trainer.model == best_model at end of training).
    trainer.save(save_path)
    with open(save_path / "train_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    log.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
