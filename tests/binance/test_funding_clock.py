"""Red tests for funding clock features.

Binance USDT-M perp funding settlement: UTC 00:00 / 08:00 / 16:00 every 8h.

Features:
  mins_to_next_funding  — minutes until next settlement, in [0, 480)
  fund_phase_sin        — sin(2π · phase), phase = minutes_into_cycle / 480
  fund_phase_cos        — cos(2π · phase)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.binance_funding_clock import funding_clock_features


def test_mins_to_next_funding_bounds_and_values():
    idx = pd.to_datetime([
        "2025-01-01 00:00:00",
        "2025-01-01 00:05:00",
        "2025-01-01 07:55:00",
        "2025-01-01 08:00:00",
        "2025-01-01 15:59:00",
        "2025-01-01 16:00:00",
        "2025-01-01 23:59:00",
    ])
    df = funding_clock_features(idx)
    m = df["mins_to_next_funding"].to_numpy()
    assert (m >= 0).all() and (m < 480).all()
    # 00:00 UTC is a settlement boundary → 0 minutes to *next* (treat 0 as "just settled", 480 to the *following*)
    # Definition: minutes remaining until the next settlement strictly after t.
    # t=00:00 → next is 08:00 → 480 minutes. But bound < 480 so we map boundary to 0.
    # Choose: mins_to_next = (480 - (t_minutes % 480)) % 480 ∈ [0, 480)
    # t=00:00 → 0
    # t=00:05 → 475
    # t=07:55 → 5
    # t=08:00 → 0
    assert m[0] == 0.0
    assert m[1] == 475.0
    assert m[2] == 5.0
    assert m[3] == 0.0


def test_fund_phase_sin_cos_unit_circle():
    idx = pd.date_range("2025-01-01", periods=96, freq="5min")  # covers one 8h cycle
    df = funding_clock_features(idx)
    s = df["fund_phase_sin"].to_numpy()
    c = df["fund_phase_cos"].to_numpy()
    # sin^2 + cos^2 = 1
    r2 = s * s + c * c
    np.testing.assert_allclose(r2, np.ones_like(r2), atol=1e-5)
    # at boundary (t=00:00), phase=0 → sin=0, cos=1
    assert abs(s[0]) < 1e-6
    assert abs(c[0] - 1.0) < 1e-6


def test_fund_phase_periodic_across_cycles():
    # 5min apart in two different cycles but same phase should produce equal sin/cos
    idx = pd.to_datetime([
        "2025-01-01 02:00:00",   # phase = 120 / 480 = 0.25
        "2025-01-01 10:00:00",   # phase = 120 / 480 = 0.25
        "2025-01-01 18:00:00",   # phase = 120 / 480 = 0.25
    ])
    df = funding_clock_features(idx)
    s = df["fund_phase_sin"].to_numpy()
    c = df["fund_phase_cos"].to_numpy()
    np.testing.assert_allclose(s, [s[0]] * 3, atol=1e-6)
    np.testing.assert_allclose(c, [c[0]] * 3, atol=1e-6)
