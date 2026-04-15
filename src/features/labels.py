"""Label computation from raw candle close prices.

벤더 캔들은 c만 의미 있으므로 c2c 계열만 생성.
"""

import numpy as np

HORIZONS = [1, 5, 10, 30, 60, 120, 240]

LABEL_NAMES = [f"c2c_{h}" for h in HORIZONS]


def compute_labels(c: np.ndarray) -> np.ndarray:
    """Compute c2c log return labels for all horizons.

    Args:
        c: close prices, float64 array of length N.

    Returns:
        (N, 7) float32 array. Columns = c2c_1, c2c_5, ..., c2c_240.
        Rows where future price is unavailable are filled with NaN.
    """
    n = len(c)
    labels = np.full((n, len(HORIZONS)), np.nan, dtype=np.float32)

    for i, h in enumerate(HORIZONS):
        if h < n:
            labels[:n - h, i] = np.log(c[h:] / (c[:n - h] + 1e-10)).astype(np.float32)

    return labels


def cross_sectional_demean(values: np.ndarray) -> np.ndarray:
    """Subtract the NaN-aware mean from a 1D array of per-stock values at one timestamp.

    Used by cs_mlp alpha_label mode: the alpha at each stock is its raw return
    minus the cross-sectional mean (which is market beta proxy) at that same
    timestamp. NaNs are preserved in the output.
    """
    arr = np.asarray(values)
    mean = np.nanmean(arr) if arr.size > 0 else 0.0
    return (arr - mean).astype(arr.dtype, copy=False)


def compute_alpha_label(c2c_120: np.ndarray, stock_indices: np.ndarray) -> np.ndarray:
    """Compute alpha label: c2c_120 - market_mean(c2c_120) per timestamp.

    Args:
        c2c_120: (N,) array of c2c_120 values for all stocks concatenated.
        stock_indices: (N,) array indicating which timestamp group each row belongs to.
            Rows with the same index are the same timestamp across stocks.

    Returns:
        (N,) float32 alpha labels.
    """
    alpha = c2c_120.copy().astype(np.float64)
    for idx in np.unique(stock_indices):
        mask = stock_indices == idx
        alpha[mask] -= np.nanmean(c2c_120[mask])
    return alpha.astype(np.float32)
