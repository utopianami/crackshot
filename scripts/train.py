"""Training script with Hydra configuration."""

import logging
import time

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.utils.config import register_configs
from src.evaluation.metrics import calculate_metrics
from src.training.registry import create_trainer

# Trainer 등록 (import 시 자동 등록)
import src.training.nn_trainer  # noqa: F401

register_configs()
log = logging.getLogger(__name__)


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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"Training model: {cfg.model.name}")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    npy_dir = cfg.nn.npy_dir
    if not npy_dir:
        raise ValueError("nn.npy_dir must be set. Run convert_kospi_npy.py first.")

    # 1. Train
    trainer = create_trainer(cfg.model.name, cfg=cfg)

    t_start = time.time()
    trainer.fit_from_npy(npy_dir)
    train_elapsed = time.time() - t_start
    log.info(f"Training time: {train_elapsed:.1f}s ({train_elapsed/60:.1f}min)")

    # 2. Evaluate (val)
    predictions = trainer._val_predictions
    actual = trainer._val_actual
    _log_eval(predictions, actual, prefix="VAL ")

    # 3. Evaluate (test)
    npy_path = Path(npy_dir)
    if (npy_path / "X_test.npy").exists():
        test_predictions, test_actual = trainer.predict_npy(npy_dir, split="test")
        _log_eval(test_predictions, test_actual, prefix="TEST ")

    # 4. Save
    save_path = Path(cfg.paths.saved_model_path) / cfg.model.name
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save(save_path)
    log.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
