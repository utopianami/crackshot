"""Label conversion utilities."""

import numpy as np


def to_classes(arr: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Continuous return → integer class label. bins = interior boundaries."""
    return np.searchsorted(bins, arr).astype(np.int32)
