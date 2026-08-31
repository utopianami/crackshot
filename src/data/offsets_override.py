"""Runtime offsets patch: reindex train/val/test boundaries by date.

Current NPY builds bake per-symbol row boundaries into offsets.json at build
time. For weekly walk-forward, we want to reuse a single NPY but shift the
split windows without rebuilding X.npy/y.npy.

The per-symbol anchor is: offsets[sym]["val_end"] row ↔ meta_test's first
timestamp for that symbol. From there, every 4h bar maps to a row delta.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# BTCUSDT is the market proxy used for fwd_res_* residualization. Including it
# in the tradable universe is redundant (its fwd_res ≈ 0 by construction) and
# it contaminates β-neutral long/short portfolios. Stripped at every offsets
# entry point so downstream code never sees it.
EXCLUDED_SYMBOLS: set[str] = {"BTCUSDT"}


def _strip_excluded(offsets: dict[str, dict]) -> dict[str, dict]:
    dropped = [s for s in offsets if s in EXCLUDED_SYMBOLS]
    if dropped:
        log.info(f"offsets filter: dropping market proxy symbols {dropped}")
    return {s: v for s, v in offsets.items() if s not in EXCLUDED_SYMBOLS}


def _parse_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None or s == "":
        return None
    return pd.Timestamp(s)


def patch_offsets(
    offsets: dict[str, dict],
    meta_test: pd.DataFrame,
    *,
    train_start: Optional[str] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    bar_hours: int = 4,
    min_train_bars: int = 0,
    require_exact_boundary: bool = True,
) -> dict[str, dict]:
    """Rewrite per-symbol offsets based on date overrides.

    Contract:
    - val_end and test_start refer to the same boundary if only one is given.
    - Boundary is row-exclusive (matches existing offsets semantics):
        train = [start, train_end)      ↔ raw period [train_start or *, val_start)
        val   = [train_end, val_end)    ↔ raw period [val_start, test_start)
        test  = [val_end, end)          ↔ raw period [test_start, test_end)
    - Unchanged fields fall back to the input offsets values.

    Skipped symbols:
    - No anchor in meta_test (no test coverage at all).
    - Post-patch window outside the symbol's [start, end] row range.
    - Post-patch train window < min_train_bars.

    Args:
        offsets: {symbol: {"start", "train_end", "val_end", "end"}}
        meta_test: [symbol, timestamp], covers original test rows.
        train_start: optional new train-window start date.
        val_start: new train→val boundary date.
        val_end: new val→test boundary date (alias of test_start).
        test_start: alias of val_end.
        test_end: new end-of-test date.
        bar_hours: bar spacing (4h default).
        min_train_bars: per-symbol minimum valid train rows after patch.

    Returns:
        New offsets dict with patched boundaries. Original keys are preserved
        for pass-through symbols; dropped symbols are omitted.
    """
    offsets = _strip_excluded(offsets)
    train_start_ts = _parse_ts(train_start)
    val_start_ts = _parse_ts(val_start)
    val_end_ts = _parse_ts(val_end)
    test_start_ts = _parse_ts(test_start)
    test_end_ts = _parse_ts(test_end)
    if val_end_ts is not None and test_start_ts is not None and val_end_ts != test_start_ts:
        raise ValueError(
            f"val_end ({val_end_ts}) and test_start ({test_start_ts}) must match"
        )
    # Single boundary name for train→val→test transitions:
    train_val_boundary = val_start_ts
    val_test_boundary = val_end_ts if val_end_ts is not None else test_start_ts

    if all(x is None for x in (train_start_ts, train_val_boundary, val_test_boundary, test_end_ts)):
        log.info("patch_offsets: no overrides given, pass-through")
        return dict(offsets)

    meta_sorted = meta_test.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    # Per-symbol ordered timestamp array spanning the ORIGINAL test window.
    # Used for exact row lookup within [orig_val_end, orig_end).
    per_sym_ts: dict[str, np.ndarray] = {
        sym: g["timestamp"].to_numpy()
        for sym, g in meta_sorted.groupby("symbol", sort=False)
    }
    bar = pd.Timedelta(hours=bar_hours)

    out: dict[str, dict] = {}
    stats = {
        "kept": 0,
        "preserved_no_anchor": 0,
        "skip_out_of_range": 0,
        "skip_short_train": 0,
        "skip_boundary_mismatch": 0,
    }

    for sym, info in offsets.items():
        start = int(info["start"])
        orig_train_end = int(info["train_end"])
        orig_val_end = int(info["val_end"])
        orig_end = int(info["end"])

        ts_arr = per_sym_ts.get(sym)
        if ts_arr is None or len(ts_arr) == 0:
            out[sym] = {"start": start, "train_end": orig_train_end,
                        "val_end": orig_val_end, "end": orig_end}
            stats["preserved_no_anchor"] += 1
            continue

        anchor_ts = pd.Timestamp(ts_arr[0])  # row = orig_val_end
        anchor_row = orig_val_end

        def _ts_to_row(ts: Optional[pd.Timestamp]) -> tuple[Optional[int], Optional[bool]]:
            """Map target ts → (row index, exact_match).

            - Within ORIGINAL test window: use meta_test row lookup (exact, gap-safe).
              Returns exact=True iff ts_arr[idx] == ts exactly (wall-clock match).
            - Before test window: formulaic anchor_row + round((ts-anchor_ts)/4h).
              Returns exact=None (unknown — no per-row timestamp metadata for train).
            - After test window: clamp to orig_end, exact=False.
            """
            if ts is None:
                return None, True
            if ts < anchor_ts:
                delta = (ts - anchor_ts) / bar
                return anchor_row + int(round(delta)), None
            idx = int(np.searchsorted(ts_arr, np.datetime64(ts), side="left"))
            if idx >= len(ts_arr):
                return orig_end, False  # beyond last test bar
            exact = pd.Timestamp(ts_arr[idx]) == ts
            return anchor_row + idx, exact

        new_start, exact_start = _ts_to_row(train_start_ts)
        new_train_end, exact_tv = _ts_to_row(train_val_boundary)
        new_val_end, exact_vt = _ts_to_row(val_test_boundary)
        new_end, exact_end = _ts_to_row(test_end_ts)

        # Strict wall-clock enforcement: drop symbols whose any checkable boundary
        # does NOT match the target wall-clock exactly. Prevents row-based rel_t
        # grouping from mixing off-by-one-bar timestamps across symbols.
        if require_exact_boundary:
            checkable = [
                (train_start_ts, exact_start),
                (train_val_boundary, exact_tv),
                (val_test_boundary, exact_vt),
                (test_end_ts, exact_end),
            ]
            mismatch = any(ts is not None and exact is False for ts, exact in checkable)
            if mismatch:
                stats["skip_boundary_mismatch"] += 1
                continue

        patched_start = start if new_start is None else new_start
        train_end = orig_train_end if new_train_end is None else new_train_end
        val_end = orig_val_end if new_val_end is None else new_val_end
        end = orig_end if new_end is None else new_end

        # Clamp to the symbol's raw-data row range.
        patched_start = max(start, min(orig_end, patched_start))
        train_end = max(patched_start, min(orig_end, train_end))
        val_end = max(train_end, min(orig_end, val_end))
        end = max(val_end, min(orig_end, end))

        if end <= val_end or val_end < train_end:
            stats["skip_out_of_range"] += 1
            continue

        if min_train_bars > 0 and (train_end - patched_start) < min_train_bars:
            stats["skip_short_train"] += 1
            continue

        out[sym] = {
            "start": int(patched_start),
            "train_end": int(train_end),
            "val_end": int(val_end),
            "end": int(end),
        }
        stats["kept"] += 1

    log.info(
        "patch_offsets: kept=%d preserved_no_anchor=%d skip_out_of_range=%d "
        "skip_short_train=%d skip_boundary_mismatch=%d "
        "(train_start=%s val_start=%s val_end=%s test_end=%s require_exact=%s)",
        stats["kept"], stats["preserved_no_anchor"], stats["skip_out_of_range"],
        stats["skip_short_train"], stats["skip_boundary_mismatch"],
        train_start_ts, train_val_boundary, val_test_boundary, test_end_ts, require_exact_boundary,
    )
    if stats["kept"] == 0:
        raise RuntimeError(f"patch_offsets kept 0 symbols; stats={stats}")
    return out


def maybe_patch_offsets_from_cfg(
    offsets: dict[str, dict],
    npy_dir,
    cfg_data,
    bar_hours: int = 4,
) -> dict[str, dict]:
    """Thin wrapper: skip if no cfg.data date overrides; else load meta_test and patch."""
    from pathlib import Path

    train_start = getattr(cfg_data, "train_start", "") or None
    val_start = getattr(cfg_data, "val_start", "") or None
    val_end = getattr(cfg_data, "val_end", "") or None
    test_start = getattr(cfg_data, "test_start", "") or None
    test_end = getattr(cfg_data, "test_end", "") or None

    if all(x in (None, "") for x in (train_start, val_start, val_end, test_start, test_end)):
        return _strip_excluded(offsets)

    npy_dir = Path(npy_dir)
    meta_test_path = npy_dir / "meta_test.parquet"
    if not meta_test_path.exists():
        raise FileNotFoundError(
            f"patch_offsets requires {meta_test_path} for per-symbol timestamp anchor"
        )
    # NPY meta.json에서 bar_hours 자동 읽기 (4h 하드코딩 → row delta 틀림 방지)
    import json as _json
    meta_path = npy_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = _json.load(open(meta_path))
            if "bar_hours" in meta:
                bar_hours = int(meta["bar_hours"])
        except Exception:
            meta = {}
    if bool(meta.get("explicit_split")):
        expected = {
            "explicit_train_start": train_start,
            "val_start": val_start,
            "val_end_ts": val_end or test_start,
            "test_start": test_start or val_end,
            "test_end": test_end,
        }
        if all(
            expected_value in (None, "")
            or str(meta.get(key, "")) == str(expected_value)
            for key, expected_value in expected.items()
        ):
            log.info("patch_offsets: input NPY declares matching explicit_split; pass-through")
            return _strip_excluded(offsets)
    meta_test = pd.read_parquet(meta_test_path)
    boundary = _parse_ts(val_end) if val_end not in (None, "") else _parse_ts(test_start)
    if boundary is not None and not meta_test.empty:
        first_ts = pd.to_datetime(
            meta_test.sort_values(["symbol", "timestamp"]).groupby("symbol")["timestamp"].first()
        )
        if len(first_ts) and (first_ts == boundary).all():
            log.info(
                "patch_offsets: input NPY already anchored at requested val/test boundary %s; pass-through",
                boundary,
            )
            return _strip_excluded(offsets)
    return patch_offsets(
        offsets,
        meta_test,
        train_start=train_start,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        bar_hours=bar_hours,
    )
