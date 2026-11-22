"""Phase 2 NPY builder — β-residual target + funding clock features.

Differences from phase1:
  * +3 features: mins_to_next_funding, fund_phase_sin, fund_phase_cos
    (total 15 features)
  * Labels: [fwd_res_1h, fwd_res_4h, fwd_res_24h, beta, vol_dummy]
    — fwd_res_* is the β-residual forward return (no BTC direction), beta is
    exposed so the portfolio loss can enforce β-neutrality.

Requires features_5m to carry beta + fwd_res_{1h,4h,24h} (run
phase2_augment_forward_residual.py first).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.binance_funding_clock import funding_clock_features


REPO_ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = REPO_ROOT / "data" / "binance" / "features_5m"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "binance" / "npy_phase2"

FEATURES_BASE = [
    "rv_4h", "amihud_4h",
    "ret_1h", "ret_4h", "ret_24h", "ret_7d",
    "res_ret_1h", "res_ret_4h", "res_ret_24h",
    "funding_mean_30d", "funding_last", "funding_sum_8h",
]
FEATURES_CLOCK = ["mins_to_next_funding", "fund_phase_sin", "fund_phase_cos"]
FEATURES = FEATURES_BASE + FEATURES_CLOCK
LABELS = ["fwd_res_1h", "fwd_res_4h", "fwd_res_24h", "beta", "vol_dummy", "fwd_funding_4h"]
FUNDING_FWD_HORIZON_HOURS = 4.0
FUNDING_PARQUET = REPO_ROOT / "data" / "binance" / "funding_rate.parquet"
PARQUET_COLS = ["open_time", "symbol"] + FEATURES_BASE + [
    "fwd_res_1h", "fwd_res_4h", "fwd_res_24h", "beta",
]
STRIDE_BARS = 48  # 4h non-overlap
TEST_DAYS = 60
VAL_DAYS = 60


def _to_utc_ns(s) -> np.ndarray:
    idx = pd.DatetimeIndex(s)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.asi8.astype("datetime64[ns]")


def _fwd_funding_sum(bar_open_times, fund_ts_ns, fund_rate, horizon_hours: float) -> np.ndarray:
    """Sum fundingRate events with bar_ts < fundingTime <= bar_ts + horizon, per bar."""
    bar_ns = _to_utc_ns(bar_open_times)
    win = np.timedelta64(int(horizon_hours * 3600 * 1_000_000_000), "ns")
    left_idx = np.searchsorted(fund_ts_ns, bar_ns, side="right")
    right_idx = np.searchsorted(fund_ts_ns, bar_ns + win, side="right")
    out = np.zeros(len(bar_ns), dtype=np.float64)
    for i in range(len(bar_ns)):
        lo, hi = left_idx[i], right_idx[i]
        if hi > lo:
            out[i] = fund_rate[lo:hi].sum()
    return out.astype(np.float32)


def _load_funding_by_symbol() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not FUNDING_PARQUET.exists():
        print(f"[warn] {FUNDING_PARQUET} not found — fwd_funding_4h will be 0")
        return {}
    df = pd.read_parquet(FUNDING_PARQUET)
    df = df.sort_values(["symbol", "fundingTime"])
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, g in df.groupby("symbol"):
        ts = _to_utc_ns(g["fundingTime"])
        rate = g["fundingRate"].to_numpy(dtype=np.float64)
        out[sym] = (ts, rate)
    print(f"[funding] loaded {len(out)} symbols, {len(df):,} events from {FUNDING_PARQUET.name}")
    return out


def _anchor_4h_utc_stride(df: pd.DataFrame, stride_bars: int = STRIDE_BARS) -> pd.DataFrame:
    """Stride 5m bars so every retained row lies on a 4h UTC boundary.

    See scripts/binance/build_npy_phase3_tierA.py for full rationale. Raw
    ``df.iloc[::stride_bars]`` is per-symbol-listing-phase which fragments
    the cross-section; anchoring on 4h UTC unifies all symbols.
    """
    ot = pd.DatetimeIndex(df["open_time"])
    if ot.tz is None:
        ot = ot.tz_localize("UTC")
    else:
        ot = ot.tz_convert("UTC")
    on_boundary = np.asarray(
        (ot.minute == 0) & (ot.second == 0) & (ot.nanosecond == 0) & (ot.hour % 4 == 0)
    )
    if not on_boundary.any():
        return df.iloc[0:0].reset_index(drop=True)
    first_idx = int(np.argmax(on_boundary))
    return df.iloc[first_idx::stride_bars].reset_index(drop=True)


def _load_all(limit: int | None, fund_by_sym: dict) -> dict[str, pd.DataFrame]:
    parts = sorted(FEAT_DIR.glob("*.parquet"))
    if limit:
        parts = parts[:limit]
    out: dict[str, pd.DataFrame] = {}
    for p in parts:
        df = pd.read_parquet(p, columns=PARQUET_COLS)
        df = df.sort_values("open_time").reset_index(drop=True)
        df = _anchor_4h_utc_stride(df, stride_bars=STRIDE_BARS)
        clock = funding_clock_features(df["open_time"].values)
        for c in FEATURES_CLOCK:
            df[c] = clock[c].to_numpy()
        sym = p.stem
        if sym in fund_by_sym:
            fts, fr = fund_by_sym[sym]
            df["fwd_funding_4h"] = _fwd_funding_sum(
                df["open_time"].values, fts, fr, FUNDING_FWD_HORIZON_HOURS,
            )
        else:
            df["fwd_funding_4h"] = 0.0
        need_cols = FEATURES + ["fwd_res_4h", "beta"]
        mask = df[need_cols].notna().all(axis=1)
        df = df[mask].reset_index(drop=True)
        out[sym] = df
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--test-end", type=str, default=None,
                   help="Override test_end (e.g. '2025-09-30'). Default: max(features).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Override output dir. Default: data/binance/npy_phase2.")
    args = p.parse_args()

    OUT_DIR = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR

    t0 = time.time()
    print(f"[load] features_5m + stride {STRIDE_BARS} ({STRIDE_BARS*5}min)", flush=True)
    fund_by_sym = _load_funding_by_symbol()
    panels = _load_all(args.limit, fund_by_sym)
    print(f"  symbols: {len(panels)}", flush=True)
    assert panels, "no symbols loaded"

    all_times = pd.concat([df["open_time"] for df in panels.values()], ignore_index=True)
    if args.test_end:
        test_end = pd.Timestamp(args.test_end)
        # Filter all panels to <= test_end
        panels = {sym: df[df["open_time"] <= test_end].reset_index(drop=True)
                  for sym, df in panels.items()}
        panels = {sym: df for sym, df in panels.items() if not df.empty}
        print(f"  filtered to test_end={test_end}, remaining symbols: {len(panels)}", flush=True)
    else:
        test_end = pd.Timestamp(all_times.max())
    test_start = test_end - pd.Timedelta(days=TEST_DAYS)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    print(f"[split] train: < {val_start}  val: [{val_start}, {test_start})  test: [{test_start}, {test_end}]", flush=True)

    total = sum(len(df) for df in panels.values())
    print(f"[alloc] total rows after stride: {total:,}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_features = len(FEATURES)
    n_labels = len(LABELS)
    X_mm = np.lib.format.open_memmap(
        OUT_DIR / "X.npy", mode="w+", dtype=np.float32, shape=(total, n_features),
    )
    y_mm = np.lib.format.open_memmap(
        OUT_DIR / "y.npy", mode="w+", dtype=np.float32, shape=(total, n_labels),
    )

    offsets: dict[str, dict] = {}
    test_rows = []
    cursor = 0
    n_train = n_val = n_test = 0

    for si, (sym, df) in enumerate(sorted(panels.items())):
        df_train = df[df["open_time"] < val_start]
        df_val = df[(df["open_time"] >= val_start) & (df["open_time"] < test_start)]
        df_test = df[df["open_time"] >= test_start]

        stock_start = cursor
        for split_name, sub in [("train", df_train), ("val", df_val), ("test", df_test)]:
            if sub.empty:
                continue
            nn = len(sub)
            xmat = sub[FEATURES].to_numpy(dtype=np.float32)
            X_mm[cursor:cursor + nn] = xmat
            ymat = np.empty((nn, n_labels), dtype=np.float32)
            ymat[:, 0] = sub["fwd_res_1h"].to_numpy(dtype=np.float32)
            ymat[:, 1] = sub["fwd_res_4h"].to_numpy(dtype=np.float32)
            ymat[:, 2] = sub["fwd_res_24h"].to_numpy(dtype=np.float32)
            ymat[:, 3] = sub["beta"].to_numpy(dtype=np.float32)
            ymat[:, 4] = 1.0  # vol dummy
            ymat[:, 5] = sub["fwd_funding_4h"].to_numpy(dtype=np.float32)
            y_mm[cursor:cursor + nn] = ymat
            cursor += nn
            if split_name == "train":
                n_train += nn
            elif split_name == "val":
                n_val += nn
            else:
                n_test += nn
                for ts in sub["open_time"]:
                    test_rows.append((sym, ts))

        train_end = stock_start + (len(df_train) if not df_train.empty else 0)
        val_end = train_end + (len(df_val) if not df_val.empty else 0)
        end = cursor
        if end > stock_start:
            offsets[sym] = {
                "start": int(stock_start),
                "train_end": int(train_end),
                "val_end": int(val_end),
                "end": int(end),
            }

        if (si + 1) % 50 == 0:
            print(f"  [{si+1}/{len(panels)}] written={cursor:,}", flush=True)

    X_mm.flush()
    y_mm.flush()
    del X_mm, y_mm

    if cursor < total:
        print(f"[trim] rewriting {cursor:,}/{total:,} rows", flush=True)
        X_full = np.load(OUT_DIR / "X.npy", mmap_mode="r")[:cursor].copy()
        y_full = np.load(OUT_DIR / "y.npy", mmap_mode="r")[:cursor].copy()
        np.save(OUT_DIR / "X.npy", X_full)
        np.save(OUT_DIR / "y.npy", y_full)

    with open(OUT_DIR / "offsets.json", "w") as f:
        json.dump(offsets, f)
    meta = {
        "n_features": n_features,
        "feature_columns": FEATURES,
        "label_columns": LABELS,
        "stride_bars": STRIDE_BARS,
        "test_days": TEST_DAYS,
        "val_days": VAL_DAYS,
        "val_start": str(val_start),
        "test_start": str(test_start),
        "test_end": str(test_end),
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if test_rows:
        pd.DataFrame(test_rows, columns=["symbol", "timestamp"]).to_parquet(
            OUT_DIR / "meta_test.parquet", index=False
        )

    print(
        f"[done] n_train={n_train:,}  n_val={n_val:,}  n_test={n_test:,}  "
        f"symbols={len(offsets)}  elapsed={(time.time()-t0)/60:.1f}m",
        flush=True,
    )
    print(f"[out] {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
