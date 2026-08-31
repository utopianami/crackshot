"""Shared guard: assert every rel_t group maps to a single wall-clock timestamp.

Used by backtest_portfolio._build_ts_groups and trainer _build_timestamp_groups
variants. The row-based right-align convention (`offset = max_len - stock_len`)
silently mismatches timestamps when symbols have gap bars mid-history — this
catches it at runtime instead of letting bad cross-sectional groups poison
scores/weights.

Only meaningful for the test split (meta_test.parquet covers those rows).
train/val splits rely on build-time NaN-free NPY + continuous anchor-:30 bars
as the guarantee.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def build_abs_to_ts(
    offsets: dict,
    orig_offsets: dict,
    npy_dir,
    split: str = "test",
) -> dict[int, object] | None:
    """Build abs_row → timestamp lookup from meta_test.parquet.

    Uses ORIGINAL offsets (pre-patch) so the index into per-symbol ts list is
    correct even if offsets.val_end/end were shifted by a runtime date override.
    Returns None if split != 'test' or meta_test.parquet missing.
    """
    if split != "test":
        return None
    mt_path = Path(npy_dir) / "meta_test.parquet"
    if not mt_path.exists():
        return None
    import pandas as pd

    mt = pd.read_parquet(mt_path)
    # Builder writes the column as "stock" (src/npy/builder.py); accept legacy
    # "symbol" too for older NPYs.
    sym_col = "stock" if "stock" in mt.columns else "symbol"
    per_sym_ts: dict[str, list] = {
        sym: g.sort_values("timestamp")["timestamp"].tolist()
        for sym, g in mt.groupby(sym_col, sort=False)
    }

    abs_to_ts: dict[int, object] = {}
    for sym, info in offsets.items():
        orig = orig_offsets.get(sym)
        if orig is None:
            continue
        orig_ve = orig["val_end"]
        orig_end = orig["end"]
        ts_list = per_sym_ts.get(sym)
        if ts_list is None:
            continue
        assert len(ts_list) == orig_end - orig_ve, (
            f"{sym}: meta_test has {len(ts_list)} rows but orig offsets say "
            f"{orig_end - orig_ve}. NPY/meta_test out of sync — rebuild."
        )
        for t in range(info["end"] - info["val_end"]):
            abs_row = info["val_end"] + t
            idx_in_meta = abs_row - orig_ve
            if 0 <= idx_in_meta < len(ts_list):
                abs_to_ts[abs_row] = ts_list[idx_in_meta]
    return abs_to_ts


def assert_single_ts_per_rel_t(
    ts_to_rows: dict[int, Iterable[int]],
    abs_to_ts: dict[int, object] | None,
) -> None:
    """Raise AssertionError if any rel_t group contains ≥2 distinct timestamps.

    Silently no-ops when abs_to_ts is None (train/val split, or no meta_test).
    """
    if not abs_to_ts:
        return
    bad: list = []
    for rel_t, rows in ts_to_rows.items():
        tss = {abs_to_ts[r] for r in rows if r in abs_to_ts}
        if len(tss) > 1:
            bad.append((rel_t, len(list(rows)), sorted(tss)[:2]))
    assert not bad, (
        f"{len(bad)} rel_t groups have mixed timestamps. First few: {bad[:3]}"
    )


def build_ts_groups_wallclock(
    offsets: dict,
    orig_offsets: dict,
    y,
    npy_dir,
    split: str,
    max_stocks: int,
    target_idx: int,
    vol_idx: int,
    for_inference: bool = False,
):
    """Wall-clock timestamp grouping for the TEST split.

    Returns (ts_rows, ts_counts, n_ts, abs_to_ts) where each group at index i
    contains rows from DIFFERENT symbols that share the SAME wall-clock. Rows
    with NaN target/vol (or vol/target == 0 when not for_inference) are filtered.

    Raises RuntimeError if split != 'test' or meta_test.parquet is missing —
    callers should fall back to row-based rel_t for train/val.
    """
    import numpy as np

    if split != "test":
        raise RuntimeError(f"wallclock grouping only supported for test split (got {split!r})")
    abs_to_ts = build_abs_to_ts(offsets, orig_offsets, npy_dir, "test")
    if abs_to_ts is None:
        raise RuntimeError(f"meta_test.parquet missing in {npy_dir}; wallclock grouping unavailable")

    ts_to_rows: dict[object, list[int]] = {}
    for sym, info in offsets.items():
        s, e = info["val_end"], info["end"]
        for abs_row in range(s, e):
            ts = abs_to_ts.get(abs_row)
            if ts is None:
                continue
            vol_val = float(y[abs_row, vol_idx])
            tgt_val = float(y[abs_row, target_idx])
            if vol_val != vol_val or tgt_val != tgt_val:
                continue
            if not for_inference and (vol_val == 0 or tgt_val == 0):
                continue
            ts_to_rows.setdefault(ts, []).append(abs_row)

    sorted_ts = sorted(ts_to_rows.keys())
    n_ts = len(sorted_ts)
    ts_rows = np.zeros((n_ts, max_stocks), dtype=np.int64)
    ts_counts = np.zeros(n_ts, dtype=np.int32)
    for i, ts in enumerate(sorted_ts):
        rows = ts_to_rows[ts]
        n = min(len(rows), max_stocks)
        ts_rows[i, :n] = rows[:n]
        ts_counts[i] = n

    # Gap-bar diagnostic: count per-symbol bars in the test window. If any
    # symbol has fewer bars than the mode, warn (helps explain missing rows
    # without breaking the run).
    import logging as _logging
    _log = _logging.getLogger(__name__)
    per_sym_bars: dict[str, int] = {}
    for sym, info in offsets.items():
        s, e = info["val_end"], info["end"]
        cnt = sum(1 for r in range(s, e) if r in abs_to_ts)
        per_sym_bars[sym] = cnt
    if per_sym_bars:
        bars_arr = np.array(list(per_sym_bars.values()))
        n_syms = len(per_sym_bars)
        mode_bars = int(np.bincount(bars_arr).argmax()) if bars_arr.size else 0
        gap_syms = int((bars_arr < mode_bars).sum())
        _log.info(
            f"[wallclock] test window: symbols={n_syms} "
            f"bars_mode={mode_bars} min={int(bars_arr.min())} max={int(bars_arr.max())} "
            f"symbols_with_gaps={gap_syms} ({100*gap_syms/n_syms:.1f}%)"
        )
    return ts_rows, ts_counts, n_ts, abs_to_ts
