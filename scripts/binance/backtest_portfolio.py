"""Backtest trained CS-MLP portfolio model on test split.

Loads best checkpoint, walks per-timestamp through test data, applies
long/short soft top-k head with ε=ε_end, computes per-bar portfolio return
(β-residual), then aggregates PnL / Sharpe / MDD.

Usage:
  PYTHONPATH=. uv run python scripts/binance/backtest_portfolio.py \
      --model-dir outputs/models/cs_mlp_portfolio_20260420_181238 \
      --data-dir data/binance/npy_phase2 \
      --split test \
      --fee-per-side 0.0005
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from types import SimpleNamespace

from src.models.architectures.cs_mlp_scorer import CrossSectionalScorerMLP
from src.training.heads.long_short_topk import long_short_soft_topk
from src.training.heads.portfolio_scorer_factory import build_portfolio_scorer


def _augment(x_raw: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return x_raw
    if mode == "rankgauss":
        from scipy.special import erfinv
        n = x_raw.shape[0]
        order = np.argsort(np.argsort(x_raw, axis=0), axis=0).astype(np.float64)
        quantiles = (order + 1.0) / (n + 1.0)
        return (np.sqrt(2.0) * erfinv(2.0 * quantiles - 1.0)).astype(np.float32)
    m = np.mean(x_raw, axis=0, keepdims=True)
    s = np.std(x_raw, axis=0, keepdims=True) + 1e-8
    z = (x_raw - m) / s
    if mode == "zscore":
        return z
    if mode == "raw_zscore":
        return np.concatenate([x_raw, z], axis=1)
    if mode == "rmz_extended":
        n = x_raw.shape[0]
        med = np.median(x_raw, axis=0, keepdims=True)
        order = np.argsort(np.argsort(x_raw, axis=0), axis=0).astype(np.float32)
        pct_rank = ((order + 0.5) / max(n, 1) - 0.5).astype(np.float32)
        return np.concatenate([
            x_raw,
            np.broadcast_to(m, x_raw.shape),
            np.broadcast_to(s, x_raw.shape),
            np.broadcast_to(med, x_raw.shape),
            pct_rank,
            z,
        ], axis=1).astype(np.float32)
    me = np.broadcast_to(m, x_raw.shape)
    return np.concatenate([x_raw, me, z], axis=1)


def _build_ts_groups(offsets: dict, y: np.ndarray, split: str, target_idx: int, vol_idx: int,
                     universe: set[str] | None = None,
                     abs_to_ts: dict | None = None,
                     meta_test=None):
    """Build ts_key → [abs_row] groups.

    For the test split (abs_to_ts available): group by wall-clock timestamp —
    the canonical, gap-safe grouping. Different symbols with different bar
    lengths still align correctly because each abs_row carries its own
    timestamp via meta_test.

    For train/val (no per-row ts metadata): fall back to right-aligned rel_t.
    Protected by the NPY build-time contract (NaN-free + continuous bars).

    Legacy path: passing meta_test is equivalent to supplying abs_to_ts.
    """
    # Build legacy abs_to_ts from meta_test (only safe when offsets unpatched).
    if abs_to_ts is None and meta_test is not None and split == "test":
        mt = meta_test.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        abs_to_ts = {}
        i = 0
        for sym in sorted(offsets.keys()):
            info = offsets[sym]
            n = info["end"] - info["val_end"]
            if n <= 0:
                continue
            for t in range(n):
                abs_row = info["val_end"] + t
                if i >= len(mt):
                    break
                if mt.iloc[i]["symbol"] != sym:
                    break
                abs_to_ts[abs_row] = mt.iloc[i]["timestamp"]
                i += 1

    # Canonical wall-clock grouping for test split.
    if split == "test" and abs_to_ts is not None:
        ts_map_wc: dict = {}
        for sym, info in offsets.items():
            if universe is not None and sym not in universe:
                continue
            s, e = info["val_end"], info["end"]
            for abs_row in range(s, e):
                ts = abs_to_ts.get(abs_row)
                if ts is None:
                    continue
                vol = float(y[abs_row, vol_idx])
                tgt = float(y[abs_row, target_idx])
                if vol != vol or tgt != tgt:
                    continue
                ts_map_wc.setdefault(ts, []).append(abs_row)
        sorted_keys = sorted(ts_map_wc.keys())
        # By construction every group contains a single wall-clock ts.
        return sorted_keys, ts_map_wc

    # Train/val (and test without abs_to_ts): row-based right-align.
    ts_map: dict[int, list[int]] = {}
    lengths = {sym: info[{"train": "train_end", "val": "val_end", "test": "end"}[split]]
               - info[{"train": "start", "val": "train_end", "test": "val_end"}[split]]
               for sym, info in offsets.items()}
    max_len = max(lengths.values())

    for sym, info in offsets.items():
        if universe is not None and sym not in universe:
            continue
        if split == "train":
            s, e = info["start"], info["train_end"]
        elif split == "val":
            s, e = info["train_end"], info["val_end"]
        else:
            s, e = info["val_end"], info["end"]
        stock_len = e - s
        offset = max_len - stock_len
        for t in range(stock_len):
            rel_t = offset + t
            abs_row = s + t
            vol = float(y[abs_row, vol_idx])
            tgt = float(y[abs_row, target_idx])
            if vol != vol or tgt != tgt:
                continue
            ts_map.setdefault(rel_t, []).append(abs_row)

    return sorted(ts_map.keys()), ts_map


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--eps", type=float, default=None, help="Override ε for head (default: eps_end from model config)")
    ap.add_argument("--fee-per-side", type=float, default=0.0005)
    ap.add_argument("--dump-weights", type=str, default=None,
                    help="If set, save per-bar per-symbol weights as parquet to this path")
    ap.add_argument("--dump-scores", type=str, default=None,
                    help="If set, save per-bar per-symbol RAW scores (pre-Sinkhorn) as parquet")
    ap.add_argument("--universe-file", type=str, default=None,
                    help="Optional txt file (one symbol per line) restricting the tradable universe")
    # Date window overrides (align with training cfg.data.*) — enables weekly walk-forward
    # without rebuilding NPY. When any of the three are set, offsets.json is patched
    # via src.data.offsets_override.patch_offsets.
    ap.add_argument("--val-start", type=str, default=None,
                    help="Train→val boundary timestamp (e.g. '2026-01-01 00:30:00')")
    ap.add_argument("--test-start", type=str, default=None,
                    help="Val→test boundary timestamp")
    ap.add_argument("--test-end", type=str, default=None,
                    help="End of test window timestamp")
    # Silent-bug guards
    ap.add_argument("--max-skip-rate", type=float, default=0.05,
                    help="Warn if fraction of skipped bars (low entity count) exceeds this.")
    args = ap.parse_args(argv)

    universe: set[str] | None = None
    if args.universe_file:
        universe = {l.strip() for l in Path(args.universe_file).read_text().splitlines() if l.strip()}
        print(f"[universe] filter loaded: {len(universe)} symbols from {args.universe_file}")

    mdir = Path(args.model_dir)
    data = Path(args.data_dir)
    cfg = json.loads((mdir / "cs_mlp_portfolio_config.json").read_text())
    meta = json.loads((data / "meta.json").read_text())
    offsets = json.loads((data / "offsets.json").read_text())
    # Strip market-proxy symbols (BTCUSDT) — they contaminate β-neutral portfolios.
    from src.data.offsets_override import EXCLUDED_SYMBOLS
    _dropped = [s for s in offsets if s in EXCLUDED_SYMBOLS]
    if _dropped:
        print(f"[universe] stripping market proxy: {_dropped}")
        offsets = {s: v for s, v in offsets.items() if s not in EXCLUDED_SYMBOLS}
    orig_offsets = {s: dict(v) for s, v in offsets.items()}  # preserve for abs_row ↔ meta_test mapping

    # Apply date-window override (same semantics as training cfg.data.*).
    # val_end is aliased to test_start here — patch_offsets enforces their equality.
    if any(v is not None for v in (args.val_start, args.test_start, args.test_end)):
        import pandas as pd
        from src.data.offsets_override import patch_offsets
        mt_for_patch = pd.read_parquet(data / "meta_test.parquet")
        offsets = patch_offsets(
            offsets, mt_for_patch,
            val_start=args.val_start,
            val_end=args.test_start,   # val_end == test_start by contract
            test_start=args.test_start,
            test_end=args.test_end,
        )
        print(f"[offsets] patched: val_start={args.val_start} test_start={args.test_start} "
              f"test_end={args.test_end} | symbols kept={len(offsets)}")

    n_feat_raw = meta["n_features"]
    mode = cfg.get("feature_mode", "raw_mean_zscore")
    aug_mult = {
        "raw": 1, "zscore": 1, "rank": 1, "rankgauss": 1,
        "raw_zscore": 2, "rmz_extended": 6,
    }.get(mode, 3)
    in_features = n_feat_raw * aug_mult

    cfg_ns = SimpleNamespace(
        hidden_units=cfg["hidden_units"],
        dropout=cfg["dropout"],
        use_layernorm=cfg.get("use_layernorm", False),
        activation=cfg.get("activation", "gelu"),
        scorer_name=cfg.get("scorer_name", "mlp"),
        tab_m_k=cfg.get("tab_m_k", 4),
        ft_t_d_token=cfg.get("ft_t_d_token", 64),
        ft_t_n_layers=cfg.get("ft_t_n_layers", 3),
        ft_t_n_heads=cfg.get("ft_t_n_heads", 4),
        ft_t_ffn_mult=cfg.get("ft_t_ffn_mult", 2),
        cs_attn_d_model=cfg.get("cs_attn_d_model", 64),
        cs_attn_n_layers=cfg.get("cs_attn_n_layers", 2),
        cs_attn_n_heads=cfg.get("cs_attn_n_heads", 4),
        cs_attn_ffn_mult=cfg.get("cs_attn_ffn_mult", 2),
        tabsyn_d_token=cfg.get("tabsyn_d_token", 32),
        tabsyn_d_latent=cfg.get("tabsyn_d_latent", 16),
        tabsyn_n_enc_layers=cfg.get("tabsyn_n_enc_layers", 2),
        tabsyn_n_dec_layers=cfg.get("tabsyn_n_dec_layers", 2),
        tabsyn_n_heads=cfg.get("tabsyn_n_heads", 4),
        tabsyn_ffn_mult=cfg.get("tabsyn_ffn_mult", 2),
        tabsyn_kl_beta=cfg.get("tabsyn_kl_beta", 1e-3),
        dcn_n_cross_layers=cfg.get("dcn_n_cross_layers", 2),
        mh_n_heads=cfg.get("mh_n_heads", 3),
        symbol_emb_dim=cfg.get("symbol_emb_dim", 32),
    )
    n_symbols_for_scorer = int(cfg.get("n_symbols", 0))
    skel = build_portfolio_scorer(cfg_ns, in_features, jax.random.PRNGKey(0),
                                  n_symbols=n_symbols_for_scorer)

    # If scorer is sym_emb, build a per-row symbol-id index that mirrors the
    # trainer's _setup_sym_emb. Each abs row in X.npy gets the integer id of
    # its symbol; rows outside any symbol slice (shouldn't exist) default to
    # the padding id (= n_symbols). At score-time we append this id as the
    # last column of x_aug so the scorer can look up the embedding.
    is_sym_emb = (cfg_ns.scorer_name == "sym_emb")
    sym_id_per_row: np.ndarray | None = None
    if is_sym_emb:
        if n_symbols_for_scorer <= 0:
            raise RuntimeError(
                "scorer_name=sym_emb but n_symbols=0 in saved config. "
                "Retrain or persist n_symbols in cs_mlp_portfolio_config.json."
            )
        x_size = int(np.load(data / "X.npy", mmap_mode="r").shape[0])
        sym_id_per_row = np.full(x_size, n_symbols_for_scorer, dtype=np.int32)
        # Prefer the saved training-time sym_to_id; only fall back to the
        # current offsets order when the saved field is absent (back-compat
        # with checkpoints written before sym_to_id persistence).
        saved_sym_to_id = cfg.get("sym_to_id") or {}
        if saved_sym_to_id:
            sym_to_id = {str(k): int(v) for k, v in saved_sym_to_id.items()}
        else:
            sym_to_id = {sym: i for i, sym in enumerate(sorted(orig_offsets.keys()))}
            print(
                "[sym_emb] WARN: saved checkpoint has no sym_to_id; rebuilding "
                "from current offsets order. Train + apply must use the same NPY."
            )
        if len(sym_to_id) != n_symbols_for_scorer:
            print(
                f"[sym_emb] WARN: sym_to_id has {len(sym_to_id)} entries but "
                f"saved n_symbols={n_symbols_for_scorer}."
            )
        unseen = [s for s in orig_offsets if s not in sym_to_id]
        if unseen:
            print(f"[sym_emb] WARN: {len(unseen)} symbols absent from trained map → padding id "
                  f"(first 5: {unseen[:5]})")
        for sym, info in orig_offsets.items():
            sid = sym_to_id.get(sym, n_symbols_for_scorer)
            sym_id_per_row[info["start"]:info["end"]] = sid
    model = eqx.tree_deserialise_leaves(str(mdir / "cs_mlp_portfolio_model.eqx"), skel)

    X = np.load(data / "X.npy", mmap_mode="r")
    y = np.load(data / "y.npy", mmap_mode="r")
    target_idx = cfg["target_idx"]
    beta_idx = cfg["beta_idx"]
    vol_idx = cfg["vol_idx"]
    eps_head = args.eps if args.eps is not None else cfg["eps_end"]

    # abs_row → (symbol, timestamp) lookup for test split.
    # Built from ORIGINAL offsets (pre-patch) so it's correct even after date override:
    # meta_test is indexed by original val_end, so abs_row - orig_val_end = position in symbol's ts list.
    abs_to_meta: dict[int, tuple] | None = None
    abs_to_ts: dict[int, object] | None = None
    if args.split == "test" and (data / "meta_test.parquet").exists():
        import pandas as pd
        mt_full = pd.read_parquet(data / "meta_test.parquet")
        per_sym_ts: dict[str, list] = {
            sym: g.sort_values("timestamp")["timestamp"].tolist()
            for sym, g in mt_full.groupby("symbol", sort=False)
        }
        abs_to_meta = {}
        abs_to_ts = {}
        for sym, info in offsets.items():
            orig_ve = orig_offsets[sym]["val_end"]
            orig_end = orig_offsets[sym]["end"]
            ts_list = per_sym_ts.get(sym)
            if ts_list is None:
                continue
            # meta_test covers [orig_ve, orig_end); its length should match.
            assert len(ts_list) == orig_end - orig_ve, (
                f"{sym}: meta_test has {len(ts_list)} rows but orig offsets say "
                f"{orig_end - orig_ve}. NPY rebuild likely out of sync with meta_test."
            )
            for t in range(info["end"] - info["val_end"]):
                abs_row = info["val_end"] + t
                idx_in_meta = abs_row - orig_ve
                if 0 <= idx_in_meta < len(ts_list):
                    ts = ts_list[idx_in_meta]
                    abs_to_meta[abs_row] = (sym, ts)
                    abs_to_ts[abs_row] = ts
        print(f"abs_row→meta lookup built: {len(abs_to_meta)} rows")

    sorted_ts, ts_map = _build_ts_groups(
        offsets, y, args.split, target_idx, vol_idx,
        universe=universe,
        abs_to_ts=abs_to_ts,
    )
    print(f"[{args.split}] timestamps: {len(sorted_ts)}")

    _is_ce = bool(getattr(model, "cross_entity_scoring", False))
    if _is_ce:
        score_fn = jax.jit(lambda xs: model(xs, key=jax.random.PRNGKey(0), inference=True))
    else:
        score_fn = jax.jit(jax.vmap(lambda xi: model(xi, key=jax.random.PRNGKey(0), inference=True)))
    head_fn = jax.jit(lambda s: long_short_soft_topk(
        s, k_long=int(cfg["k_long"]), k_short=int(cfg["k_short"]),
        eps=float(eps_head), n_iter=int(cfg["sinkhorn_iter"]),
    ).w)

    returns: list[float] = []
    turnovers: list[float] = []
    net_betas: list[float] = []
    min_stocks = max(int(cfg["k_long"]), int(cfg["k_short"])) * 2 + 2
    # Each symbol owns a contiguous [start, end) block of absolute NPY rows.
    # Use that block's slot for the lifetime of a position, including on exit.
    symbol_ends = np.sort([info["end"] for info in orig_offsets.values()])
    previous_positions = np.zeros(len(symbol_ends), dtype=np.float64)
    evaluated_times = []
    skipped = 0
    weight_records: list[dict] = []
    score_records: list[dict] = []
    for rel_t in sorted_ts:
        rows = ts_map[rel_t]
        if len(rows) < min_stocks:
            skipped += 1
            continue
        x_raw = np.asarray(X[rows], dtype=np.float32)
        r = np.asarray(y[rows, target_idx], dtype=np.float32)
        betas = np.asarray(y[rows, beta_idx], dtype=np.float32)
        # Silent-bug guard: NPY must be NaN-free (anchor30 build asserts this).
        # If we see NaN here, the build is broken — abort loud, don't mask with nan_to_num.
        if np.isnan(x_raw).any() or np.isnan(r).any() or np.isnan(betas).any():
            raise RuntimeError(
                f"NaN at rel_t={rel_t}: X={int(np.isnan(x_raw).sum())}, "
                f"r={int(np.isnan(r).sum())}, beta={int(np.isnan(betas).sum())}. "
                "NPY build is broken — rebuild with assert-no-NaN path."
            )
        x_aug = _augment(x_raw, mode)
        if sym_id_per_row is not None:
            sym_col = sym_id_per_row[rows].astype(np.float32)[:, None]
            x_aug = np.concatenate([x_aug, sym_col], axis=1)
        scores = np.asarray(score_fn(jnp.asarray(x_aug)))
        w = np.asarray(head_fn(jnp.asarray(scores)))
        port_ret = float(np.sum(w * r))
        net_beta = float(np.sum(w * betas))
        positions = np.zeros_like(previous_positions)
        positions[np.searchsorted(symbol_ends, rows, side="right")] = w
        turn_l1 = float(np.abs(positions - previous_positions).sum())
        previous_positions = positions
        evaluated_times.append(rel_t)
        returns.append(port_ret)
        turnovers.append(turn_l1)
        net_betas.append(net_beta)
        if abs_to_meta is not None:
            for abs_row, wi, sc in zip(rows, w, scores):
                info = abs_to_meta.get(abs_row)
                if info is None:
                    continue
                if float(wi) != 0.0:
                    weight_records.append({
                        "rel_t": rel_t,
                        "symbol": info[0],
                        "timestamp": info[1],
                        "weight": float(wi),
                    })
                if args.dump_scores:
                    score_records.append({
                        "rel_t": rel_t,
                        "symbol": info[0],
                        "timestamp": info[1],
                        "score": float(sc),
                    })

    if not evaluated_times:
        raise ValueError(f"No eligible {args.split} bars with at least {min_stocks} assets.")
    returns = np.asarray(returns)
    turnovers = np.asarray(turnovers)
    net_betas = np.asarray(net_betas)

    gross = returns.copy()
    net = returns - turnovers * args.fee_per_side
    cum_gross = np.cumsum(gross)
    cum_net = np.cumsum(net)

    # Stats (per-bar = 4h)
    n = len(returns)
    bars_per_year = 24 / 4 * 365  # 2190
    mean_g, std_g = gross.mean(), gross.std() + 1e-12
    mean_n, std_n = net.mean(), net.std() + 1e-12
    sharpe_g = mean_g / std_g * np.sqrt(bars_per_year)
    sharpe_n = mean_n / std_n * np.sqrt(bars_per_year)

    def maxdd(curve):
        peak = np.maximum.accumulate(curve)
        return float((curve - peak).min())

    # Skip-rate guard: if too many bars dropped for low-entity count, scores are unreliable.
    total_bars = len(sorted_ts)
    if total_bars > 0:
        skip_rate = skipped / total_bars
        if skip_rate > args.max_skip_rate:
            print(f"[WARN] high skip rate: {skipped}/{total_bars} = {skip_rate:.1%} "
                  f"(> {args.max_skip_rate:.1%}). min_stocks={min_stocks} might be too high "
                  f"or universe too sparse.")
    print(f"\n=== BACKTEST {args.split} (n={n} bars, 4h each, ≈{n/6:.1f} days, skipped={skipped} low-entity bars) ===")
    print(f"Gross PnL: total={cum_gross[-1]*100:.2f}%, mean={mean_g*100:.4f}%/bar, std={std_g*100:.4f}%")
    print(f"Net   PnL: total={cum_net[-1]*100:.2f}%, mean={mean_n*100:.4f}%/bar, std={std_n*100:.4f}%")
    print(f"Sharpe (annualized): gross={sharpe_g:.2f}, net={sharpe_n:.2f}")
    print(f"MaxDD: gross={maxdd(cum_gross)*100:.2f}%, net={maxdd(cum_net)*100:.2f}%")
    print(f"Avg turnover/bar (L1): {turnovers.mean():.2f}")
    print(f"Avg net β exposure: {net_betas.mean():.3f} (std={net_betas.std():.3f})")
    print(f"Hit rate (positive bars): gross={(gross>0).mean()*100:.1f}%, net={(net>0).mean()*100:.1f}%")

    # Save series
    out_csv = mdir / f"backtest_{args.split}.csv"
    import pandas as pd
    pd.DataFrame({
        "rel_t": evaluated_times,
        "port_ret": gross,
        "net_ret": net,
        "turnover_l1": turnovers,
        "net_beta": net_betas,
        "cum_gross": cum_gross,
        "cum_net": cum_net,
    }).to_csv(out_csv, index=False)
    print(f"\nSaved series: {out_csv}")

    if args.dump_weights:
        import pandas as pd
        wdf = pd.DataFrame(weight_records)
        Path(args.dump_weights).parent.mkdir(parents=True, exist_ok=True)
        wdf.to_parquet(args.dump_weights, index=False)
        print(f"Saved weights: {args.dump_weights} ({len(wdf)} rows)")
    if args.dump_scores:
        import pandas as pd
        sdf = pd.DataFrame(score_records)
        Path(args.dump_scores).parent.mkdir(parents=True, exist_ok=True)
        sdf.to_parquet(args.dump_scores, index=False)
        print(f"Saved scores: {args.dump_scores} ({len(sdf)} rows)")


if __name__ == "__main__":
    sys.exit(main())
