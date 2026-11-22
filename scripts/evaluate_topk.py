"""Top-k evaluation utilities for classification models.

Used by scripts/train.py for inline evaluation during training.
"""

import numpy as np


def compute_score(probs: np.ndarray, weighted: bool = False) -> np.ndarray:
    """Compute prediction score from class probabilities.

    Default: Score = (P[3] + P[4]) - (P[0] + P[1])
    Weighted: Score = (P[3] + 2*P[4]) - (P[1] + 2*P[0])
      → extreme classes weighted 2x to favor magnitude
    """
    if probs.ndim == 1:
        raise ValueError("probs must be 2D array of shape (N, num_classes)")
    num_classes = probs.shape[1]
    if num_classes == 5:
        if weighted:
            return (probs[:, 3] + 2 * probs[:, 4]) - (probs[:, 1] + 2 * probs[:, 0])
        return (probs[:, 3] + probs[:, 4]) - (probs[:, 0] + probs[:, 1])
    elif num_classes == 2:
        return probs[:, 1] - probs[:, 0]
    else:
        mid = num_classes // 2
        return probs[:, mid:].sum(axis=1) - probs[:, :mid].sum(axis=1)


def topk_stats(score: np.ndarray, actual: np.ndarray, k_pct: float):
    """Compute direction accuracy and mean return for top/bottom k%."""
    n = len(score)
    n_k = max(1, int(round(n * k_pct / 100)))

    # Top k% (highest score -> bullish signals)
    top_idx = np.argpartition(score, -n_k)[-n_k:]
    top_actual = actual[top_idx]
    top_dir_acc = float(np.mean(top_actual > 0)) * 100
    top_mean_ret = float(np.mean(top_actual)) * 100  # as percentage

    # Bottom k% (lowest score -> bearish signals)
    bot_idx = np.argpartition(score, n_k - 1)[:n_k]
    bot_actual = actual[bot_idx]
    bot_dir_acc = float(np.mean(bot_actual < 0)) * 100
    bot_mean_ret = float(np.mean(bot_actual)) * 100  # as percentage

    return {
        "top_dir_acc": top_dir_acc,
        "top_mean_ret": top_mean_ret,
        "bot_dir_acc": bot_dir_acc,
        "bot_mean_ret": bot_mean_ret,
        "n_top": n_k,
        "n_bot": n_k,
        "n_total": n,
    }
