"""Training script with OmegaConf configuration.

Usage:
    uv run python -m scripts.train -c configs/tab_m_binance_phase1.yaml data.data_path=/path/to/npy

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
import json
import logging
import time

import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

from scripts.evaluate_topk import compute_score, topk_stats
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
    score = compute_score(probs)
    # Filter vol==0 and c2c_120==0
    mask = actual != 0
    score_f, actual_f = score[mask], actual[mask]
    log.info(f"=== {prefix}Classification Eval (N={len(actual_f):,}, filtered {(~mask).sum():,} zero-label rows) ===")
    if not len(actual_f):
        return
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("-c", "--config", type=str, default=None, help="YAML config file")
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Explicit run folder name (overrides default {model}_{ts_suffix}). "
             "Used for both saved_model_path and inference_path subdir.",
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value dotlist)")
    return parser


def _resolve_run_name(run_name: str | None, model_name: str, ts_suffix: str) -> str:
    """Return run_name if non-empty, else fall back to default `{model}_{ts}`."""
    if run_name:
        return run_name
    return f"{model_name}_{ts_suffix}"


def main():
    args = _build_arg_parser().parse_args()

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
        raise ValueError("Set data.data_path to the experiment NPY directory.")

    # Timestamp suffix for unique output paths (used for both model save and predictions)
    # Include microseconds + PID so parallel launches within the same second don't collide.
    from datetime import datetime
    ts_suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_p{os.getpid()}"

    # 1. Train
    trainer = create_trainer(model_name, cfg=cfg)

    # Inject save path so the trainer can overwrite-save "best" on every val
    # improvement (crash-safe). If training finishes cleanly, trainer.save()
    # is called again at the end below, which is a no-op rewrite of the same
    # best weights.
    run_name = _resolve_run_name(args.run_name, model_name, ts_suffix)
    save_path = Path(cfg.paths.saved_model_path) / run_name
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
    # offsets에 test rows 없으면 (val_end == end 모두) test eval 스킵 — --no-test 빌드.
    if has_test and (npy_path / "offsets.json").exists():
        try:
            _o = json.loads((npy_path / "offsets.json").read_text())
            if not any(int(v.get("end", 0)) > int(v.get("val_end", 0)) for v in _o.values()):
                has_test = False
                log.info("No test rows in offsets — skipping test eval.")
        except Exception:
            pass
    if has_test:
        test_predictions, test_actual = trainer.predict_npy(npy_dir, split="test")
        if is_classification(model_name):
            _log_eval_cls(test_predictions, test_actual, prefix="TEST ")
            # Save predictions for evaluate_topk.py
            out_dir = Path(cfg.paths.inference_path) / run_name
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
