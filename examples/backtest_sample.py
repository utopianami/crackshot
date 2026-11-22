"""Run the Phase 2 builder and backtester on a small synthetic feature panel."""

import json
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import pandas as pd

from scripts.binance import backtest_portfolio, build_npy_phase2
from src.models.architectures.cs_mlp_scorer import CrossSectionalScorerMLP


SAMPLE = Path(__file__).parent / "data" / "portfolio_features.csv"


def run_sample(directory: Path) -> None:
    print(f"Checking backtest accounting with synthetic observations from {SAMPLE.name}", flush=True)
    features = pd.read_csv(SAMPLE, parse_dates=["open_time"])
    feature_dir = directory / "features_5m"
    feature_dir.mkdir()
    for symbol, rows in features.groupby("symbol"):
        rows.to_parquet(feature_dir / f"{symbol}.parquet", index=False)

    arrays = directory / "arrays"
    build_npy_phase2.main(["--source-dir", str(directory), "--out-dir", str(arrays)])

    # A seeded linear scorer exercises checkpoint loading without private weights.
    model_dir = directory / "model"
    model_dir.mkdir()
    meta = json.loads((arrays / "meta.json").read_text())
    model = CrossSectionalScorerMLP(
        meta["n_features"], hidden_units=[], dropout=0.0, key=jax.random.PRNGKey(42)
    )
    eqx.tree_serialise_leaves(model_dir / "cs_mlp_portfolio_model.eqx", model)
    config = {
        "hidden_units": [],
        "dropout": 0.0,
        "feature_mode": "zscore",
        "k_long": 1,
        "k_short": 1,
        "eps_end": 0.2,
        "sinkhorn_iter": 100,
        "target_idx": meta["label_columns"].index("fwd_res_4h"),
        "beta_idx": meta["label_columns"].index("beta"),
        "vol_idx": meta["label_columns"].index("vol_dummy"),
    }
    (model_dir / "cs_mlp_portfolio_config.json").write_text(json.dumps(config))
    backtest_portfolio.main([
        "--model-dir", str(model_dir), "--data-dir", str(arrays),
        "--fee-per-side", "0.0005", "--dump-weights", str(directory / "weights.parquet"),
    ])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        run_sample(workspace)
        result = pd.read_csv(workspace / "model" / "backtest_test.csv")
        print("\nSynthetic example · evaluated timestamps and L1 turnover")
        print(result[["rel_t", "turnover_l1"]].to_string(index=False))
