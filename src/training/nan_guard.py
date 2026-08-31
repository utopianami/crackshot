"""Silent-NaN guard helpers.

Trainer guards against NaN in training data. The y memmap has NaN tails per
split per stock by design (`compute_labels` produces forward-return labels;
the last `max_horizon` rows of each split per stock have no future and stay
NaN). The guard must therefore check only the rows actually used for
training (`tr_indices` / `va_indices` after `build_splits`), not the full
memmap — otherwise it fires on the by-design NaN tail.

See:
  - `src/features/labels.py:compute_labels` (NaN tail by design)
  - `src/data/split.py:build_splits` (filters NaN rows from indices)
  - `BRAIN/kospi/dont_repeat.md` "no_silent_nan_imputation"
"""
from __future__ import annotations

import numpy as np


def has_nan_at(
    arr: np.ndarray,
    indices: np.ndarray,
    chunk: int = 10_000_000,
) -> bool:
    """True if any NaN exists in `arr[indices]`. Streams in chunks.

    Short-circuits at first NaN. `chunk` caps temporary materialization
    (10M × 7 cols × 4B ≈ 280MB).
    """
    if len(indices) == 0:
        return False
    for i in range(0, len(indices), chunk):
        if np.isnan(arr[indices[i:i + chunk]]).any():
            return True
    return False
