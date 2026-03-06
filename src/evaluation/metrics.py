"""Prediction quality metrics for regression models.

Measures how well model predictions match actual direction and magnitude:
- Direction accuracy (overall, and by actual direction)
- Top/Bottom k% precision (strongest predictions)
- Rank correlation (Spearman)
- RMSE, MAE
"""

import numpy as np
from scipy import stats


def direction_accuracy(predictions: np.ndarray, actual: np.ndarray) -> dict:
    """Calculate direction accuracy metrics.

    Args:
        predictions: Model predictions
        actual: Actual values

    Returns:
        Dict with overall, up, and down direction accuracy
    """
    pred_up = predictions > 0
    actual_up = actual > 0

    # Overall direction accuracy
    correct = pred_up == actual_up
    overall = float(np.mean(correct))

    # When model predicts UP, how often was it actually up?
    pred_up_mask = pred_up
    precision_up = float(np.mean(actual[pred_up_mask] > 0)) if pred_up_mask.sum() > 0 else 0.0

    # When model predicts DOWN, how often was it actually down?
    pred_down_mask = ~pred_up
    precision_down = float(np.mean(actual[pred_down_mask] < 0)) if pred_down_mask.sum() > 0 else 0.0

    return {
        "direction_accuracy": overall,
        "precision_up": precision_up,
        "precision_down": precision_down,
        "n_pred_up": int(pred_up_mask.sum()),
        "n_pred_down": int(pred_down_mask.sum()),
    }


def topk_accuracy(predictions: np.ndarray, actual: np.ndarray, k_pcts: list[float] = [0.1, 0.5, 1, 5, 10]) -> dict:
    """Calculate accuracy for top/bottom k% of predictions.

    For the strongest buy signals (top k% of predictions), what fraction actually went up?
    For the strongest sell signals (bottom k% of predictions), what fraction actually went down?

    Args:
        predictions: Model predictions
        actual: Actual values
        k_pcts: List of percentages to evaluate

    Returns:
        Dict with top-k and bottom-k accuracy for each k
    """
    n = len(predictions)
    results = {}

    for k in k_pcts:
        n_k = max(1, int(n * k / 100))

        # Top k% (strongest buy signals)
        top_idx = np.argpartition(predictions, -n_k)[-n_k:]
        top_actual = actual[top_idx]
        top_acc = float(np.mean(top_actual > 0))
        top_mean_return = float(np.mean(top_actual))

        # Bottom k% (strongest sell signals)
        bot_idx = np.argpartition(predictions, n_k)[:n_k]
        bot_actual = actual[bot_idx]
        bot_acc = float(np.mean(bot_actual < 0))
        bot_mean_return = float(np.mean(bot_actual))

        results[f"top{k}pct_acc"] = top_acc
        results[f"top{k}pct_mean_return"] = top_mean_return
        results[f"bot{k}pct_acc"] = bot_acc
        results[f"bot{k}pct_mean_return"] = bot_mean_return

    return results


def regression_metrics(predictions: np.ndarray, actual: np.ndarray) -> dict:
    """Calculate standard regression metrics.

    Args:
        predictions: Model predictions
        actual: Actual values

    Returns:
        Dict with RMSE, MAE, and Spearman correlation
    """
    rmse = float(np.sqrt(np.mean((predictions - actual) ** 2)))
    mae = float(np.mean(np.abs(predictions - actual)))

    if np.std(predictions) < 1e-12:
        spearman = 0.0
        pearson = 0.0
    else:
        spearman = float(stats.spearmanr(predictions, actual).statistic)
        pearson = float(stats.pearsonr(predictions, actual).statistic)

    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_corr": pearson,
        "spearman_corr": spearman,
    }


def calculate_metrics(predictions: np.ndarray, actual: np.ndarray) -> dict:
    """Calculate all prediction quality metrics.

    Args:
        predictions: Model predictions
        actual: Actual values

    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    metrics.update(regression_metrics(predictions, actual))
    metrics.update(direction_accuracy(predictions, actual))
    metrics.update(topk_accuracy(predictions, actual, k_pcts=[0.1, 0.5, 1, 5, 10]))
    return metrics
