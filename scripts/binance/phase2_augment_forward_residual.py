"""Append forward β-residual return columns + β(t) to features_5m parquets.

Mirrors phase0_add_btc_residual.py but for the TRAINING TARGET:
  fwd_res_H(t) = sum_{k=t+1..t+H}(r_coin(k) - β(t) * r_btc(k))
  beta         = rolling OLS(coin on BTC, 7d window)

These columns power the differentiable portfolio loss (Phase 2-R): the
β column feeds the β-neutrality constraint, fwd_res is the residual
forward return the loss rewards.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.binance_btc_residual import compute_forward_residual_returns


REPO_ROOT = Path(__file__).resolve().parents[2]
KLINES_DIR = Path(__file__).resolve().parents[2] / "data" / "binance" / "klines_5m"
FEAT_DIR = KLINES_DIR.parent / "features_5m"

HORIZONS = {"1h": 12, "4h": 48, "24h": 288}
BETA_WINDOW = 2016  # 7d × 288


def _load_symbol_klines(symbol: str) -> pd.DataFrame:
    parts = sorted((KLINES_DIR / symbol).glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    return df


def _one_bar_returns(klines: pd.DataFrame) -> pd.Series:
    close = klines["close"].astype(float).values
    r = np.full(len(close), np.nan)
    r[1:] = close[1:] / close[:-1] - 1.0
    return pd.Series(r, index=klines["open_time"].values, name="ret_1bar")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--symbols", type=str, default=None)
    args = p.parse_args()

    t0 = time.time()
    print(f"[start] loading BTCUSDT 1-bar returns", flush=True)
    btc_klines = _load_symbol_klines("BTCUSDT")
    btc_ret = _one_bar_returns(btc_klines)
    print(f"  BTC 1-bar: n={len(btc_ret):,}", flush=True)

    all_syms = sorted([p.stem for p in FEAT_DIR.glob("*.parquet")])
    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        syms = all_syms
    if args.limit:
        syms = syms[: args.limit]

    print(f"[plan] N={len(syms)}  out={FEAT_DIR}", flush=True)
    stats = {"ok": 0, "no_klines": 0, "err": 0}
    for i, sym in enumerate(syms, 1):
        try:
            feat_path = FEAT_DIR / f"{sym}.parquet"
            feat = pd.read_parquet(feat_path)
            klines = _load_symbol_klines(sym)
            if klines.empty:
                stats["no_klines"] += 1
                continue
            coin_ret = _one_bar_returns(klines)

            idx = pd.Index(feat["open_time"].values)
            coin_aligned = coin_ret.reindex(idx).to_numpy(dtype=float)
            btc_aligned = btc_ret.reindex(idx).to_numpy(dtype=float)

            fwd = compute_forward_residual_returns(
                coin_1bar=coin_aligned,
                btc_1bar=btc_aligned,
                horizons=HORIZONS,
                beta_window=BETA_WINDOW,
            )
            feat["beta"] = fwd["beta"].to_numpy()
            for col in ["fwd_res_1h", "fwd_res_4h", "fwd_res_24h"]:
                feat[col] = fwd[col].to_numpy()
            feat.to_parquet(feat_path, index=False)
            stats["ok"] += 1
        except Exception as e:
            stats["err"] += 1
            print(f"[err] {sym}: {type(e).__name__}: {e}", flush=True)
        if i % 25 == 0 or i == len(syms):
            el = time.time() - t0
            print(
                f"[{i}/{len(syms)}] ok={stats['ok']} no_klines={stats['no_klines']} "
                f"err={stats['err']} elapsed={el/60:.1f}m",
                flush=True,
            )

    print(f"[done] {stats} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    return 0 if stats["err"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
