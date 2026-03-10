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
import src.training.ae_mlp_trainer  # noqa: F401
import src.training.mamba_tab_trainer  # noqa: F401
import src.training.tsc_mamba_trainer  # noqa: F401
import src.training.tab_m_trainer  # noqa: F401

register_configs()
log = logging.getLogger(__name__)


def _log_eval_cls(probs, actual, prefix=""):
    """Log evaluation for classification models (probs [N,5], actual [N] raw returns)."""
    from scripts.evaluate_topk import compute_score, topk_stats
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"Training model: {cfg.model.name}")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Resolve npy_dir based on model type
    if cfg.model.name == "ae_mlp":
        npy_dir = cfg.ae_mlp.npy_dir
    elif cfg.model.name == "mamba_tab":
        npy_dir = cfg.mamba_tab.npy_dir
    elif cfg.model.name == "tsc_mamba":
        npy_dir = cfg.tsc_mamba.npy_dir
    elif cfg.model.name == "tab_m":
        npy_dir = cfg.tab_m.npy_dir
    else:
        npy_dir = cfg.nn.npy_dir
    if not npy_dir:
        raise ValueError("npy_dir must be set. Run convert_kospi_npy.py first.")

    # 1. Train
    trainer = create_trainer(cfg.model.name, cfg=cfg)

    t_start = time.time()
    trainer.fit_from_npy(npy_dir)
    train_elapsed = time.time() - t_start
    log.info(f"Training time: {train_elapsed:.1f}s ({train_elapsed/60:.1f}min)")

    # 2. Evaluate (val)
    predictions = trainer._val_predictions
    actual = trainer._val_actual
    if cfg.model.name in ("ae_mlp", "mamba_tab", "tsc_mamba", "tab_m"):
        # classification models return (probs [N,5], raw_c2c_120 [N])
        _log_eval_cls(predictions, actual, prefix="VAL ")
    else:
        _log_eval(predictions, actual, prefix="VAL ")

    # 3. Evaluate (test)
    npy_path = Path(npy_dir)
    has_test = (npy_path / "X_test.npy").exists() or (npy_path / "X.npy").exists()
    if has_test:
        test_predictions, test_actual = trainer.predict_npy(npy_dir, split="test")
        if cfg.model.name in ("ae_mlp", "mamba_tab", "tsc_mamba", "tab_m"):
            _log_eval_cls(test_predictions, test_actual, prefix="TEST ")
            # Save predictions for evaluate_topk.py
            out_dir = Path(cfg.paths.inference_path) / cfg.model.name
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "test_probs.npy", test_predictions)
            np.save(out_dir / "test_actual.npy", test_actual)
            log.info(f"Test predictions saved to {out_dir}")
        else:
            _log_eval(test_predictions, test_actual, prefix="TEST ")

    # 4. Save
    save_path = Path(cfg.paths.saved_model_path) / cfg.model.name
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save(save_path)
    log.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
