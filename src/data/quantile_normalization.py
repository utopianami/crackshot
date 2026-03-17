"""Quantile normalization: feature 값을 quantile rank로 변환."""

import numpy as np


def fit_quantile_params(
    X: np.ndarray,
    indices: np.ndarray,
    n_quantiles: int = 1000,
    output_distribution: str = "uniform",
    max_samples: int = 2_000_000,
    seed: int = 42,
    feat_min: np.ndarray | None = None,
    feat_max: np.ndarray | None = None,
) -> dict:
    """학습 데이터에서 quantile 참조 테이블 생성.

    Args:
        X: Feature array (memmap OK), shape (N, F).
        indices: Valid training row indices.
        n_quantiles: Number of quantile landmarks.
        output_distribution: "uniform" ([0,1]) or "normal" (standard normal).
        max_samples: Subsample if len(indices) exceeds this.
        seed: Random seed for subsampling.
        feat_min: Pre-computed per-feature min from full training set.
        feat_max: Pre-computed per-feature max from full training set.

    Returns:
        Scaler params dict with mode="quantile".
    """
    rng = np.random.default_rng(seed)
    if len(indices) > max_samples:
        sample_idx = rng.choice(indices, max_samples, replace=False)
    else:
        sample_idx = indices

    X_sample = np.array(X[sample_idx], dtype=np.float64)
    quantiles = np.linspace(0, 1, n_quantiles)
    references = np.quantile(X_sample, quantiles, axis=0)  # (n_quantiles, F)

    # Use full training set min/max for endpoints
    if feat_min is not None:
        references[0] = feat_min.astype(np.float64)
    if feat_max is not None:
        references[-1] = feat_max.astype(np.float64)

    return {
        "mode": "quantile",
        "references": references.astype(np.float64),
        "quantiles": quantiles,
        "output_distribution": output_distribution,
    }


def _quantile_transform(X: np.ndarray, params: dict) -> np.ndarray:
    """Quantile transform using pre-fitted references."""
    references = params["references"]  # (n_quantiles, F)
    quantiles = params["quantiles"]    # (n_quantiles,)
    output_dist = params.get("output_distribution", "uniform")

    X_f64 = X.astype(np.float64)
    result = np.empty_like(X, dtype=np.float32)

    for j in range(X.shape[-1]):
        ref_col = references[:, j]
        result[..., j] = np.interp(X_f64[..., j], ref_col, quantiles).astype(np.float32)

    if output_dist == "normal":
        from scipy.stats import norm
        result = np.clip(result, 1e-7, 1 - 1e-7)
        result = norm.ppf(result).astype(np.float32)

    return result


def transform_np(X: np.ndarray, params: dict) -> np.ndarray:
    """Quantile normalization 적용. quantile mode만 지원."""
    mode = params.get("mode")
    if mode != "quantile":
        raise ValueError(f"Unsupported mode: {mode!r}. Only 'quantile' is supported.")
    return _quantile_transform(X, params)
