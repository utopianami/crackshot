"""Funding clock features for Binance USDT-M perpetuals.

Binance settles perp funding every 8 hours at UTC 00:00, 08:00, 16:00.

Features (per bar):
  mins_to_next_funding  ∈ [0, 480)  minutes until the next settlement (0 at boundary)
  fund_phase_sin        sin(2π · phase)    phase = (mins_into_cycle) / 480
  fund_phase_cos        cos(2π · phase)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CYCLE_MIN = 8 * 60  # 480


def funding_clock_features(timestamps) -> pd.DataFrame:
    """Compute the 3 funding clock features for a datetime index."""
    ts = pd.to_datetime(timestamps)
    total_mins = ts.hour * 60 + ts.minute
    mins_into_cycle = total_mins % CYCLE_MIN
    # Minutes remaining until the next settlement. At the boundary (mins_into_cycle==0),
    # we use 0 to keep the feature in [0, 480).
    mins_to_next = np.where(
        mins_into_cycle == 0, 0, CYCLE_MIN - mins_into_cycle
    ).astype(np.float32)

    phase = mins_into_cycle / CYCLE_MIN  # [0, 1)
    s = np.sin(2 * np.pi * phase).astype(np.float32)
    c = np.cos(2 * np.pi * phase).astype(np.float32)

    return pd.DataFrame(
        {
            "mins_to_next_funding": mins_to_next,
            "fund_phase_sin": s,
            "fund_phase_cos": c,
        }
    )
