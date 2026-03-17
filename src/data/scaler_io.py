"""Scaler parameter save/load (quantile mode only)."""

from pathlib import Path

import numpy as np


def save_scaler_params(path: Path, params: dict) -> None:
    """Save quantile scaler params to npz."""
    np.savez(
        path,
        mode=np.array("quantile"),
        references=params["references"],
        quantiles=params["quantiles"],
        output_distribution=np.array(params["output_distribution"]),
    )


def load_scaler_params(path: Path) -> dict:
    """Load quantile scaler params from npz."""
    data = np.load(path)
    return {
        "mode": "quantile",
        "references": data["references"],
        "quantiles": data["quantiles"],
        "output_distribution": str(data["output_distribution"]),
    }
