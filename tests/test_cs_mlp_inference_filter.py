"""Regression test: _build_timestamp_groups must NOT filter target==0 rows at inference.

Bug (2026-04-06, dont_repeat.md): predict_npy was filtering rows with vol==0 or target==0
  at inference time, then filling them with uniform probs [0.2,...]. Reference backtests got
  2568/14060 timestamps with uniform predictions.
Fix: inference path must include all rows where features are valid (drop only NaN).
"""

import json
from pathlib import Path
import numpy as np
import pytest

from omegaconf import OmegaConf
from src.config import load_config
from src.training.registry import create_trainer, discover_trainers

discover_trainers()


@pytest.fixture
def tiny_npy(tmp_path):
    """Create a tiny NPY with known target==0 rows."""
    npy_dir = tmp_path / "npy"
    npy_dir.mkdir()
    # 2 stocks × 4 bars, 100 features (dummy)
    n_rows = 8
    n_feat = 100
    X = np.random.randn(n_rows, n_feat).astype(np.float32)
    # y has 8 label cols (7 c2c horizons + 1 vol). Layout: [c2c_1, c2c_5, c2c_10, c2c_30, c2c_60, c2c_120, c2c_240, vol]
    y = np.ones((n_rows, 8), dtype=np.float32) * 0.01  # all non-zero small values
    y[:, -1] = 1.0  # vol = 1 everywhere
    # Set row 2 target (c2c_120 = idx 5) to 0 — this should NOT be filtered at inference
    y[2, 5] = 0.0

    np.save(npy_dir / "X.npy", X)
    np.save(npy_dir / "y.npy", y)
    offsets = {
        "AAA": {"start": 0, "train_end": 2, "val_end": 3, "end": 4},
        "BBB": {"start": 4, "train_end": 6, "val_end": 7, "end": 8},
    }
    with open(npy_dir / "offsets.json", "w") as f:
        json.dump(offsets, f)

    # meta_test.parquet required by the pipeline? create minimal
    import pandas as pd
    meta_test = pd.DataFrame({
        "stock": ["AAA"] * 1 + ["BBB"] * 1,
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:00:00"],
    })
    meta_test.to_parquet(npy_dir / "meta_test.parquet")

    return npy_dir


def test_inference_does_not_filter_zero_target(tiny_npy):
    """At inference, a row where c2c_120==0 must get a NON-uniform prediction
    (actual model output), not the uniform fallback [0.2, 0.2, 0.2, 0.2, 0.2]."""
    cfg = load_config("configs/cs_mlp_200_base.yaml")
    cfg.data.data_path = str(tiny_npy)
    trainer = create_trainer("cs_mlp", cfg=cfg)
    # Manually initialize the model without training (tiny NPY has no valid train rows)
    from src.models.architectures.cs_mlp import CrossSectionalMLP
    import jax
    trainer.model = CrossSectionalMLP(
        in_features=300,  # 100 * 3 for raw_mean_zscore
        hidden_units=[256, 128],
        dropout=0.4,
        num_classes=5,
        num_horizons=3,
        key=jax.random.PRNGKey(0),
    )
    trainer.feature_columns = [f"base_{i}" for i in range(100)]
    trainer._feature_mode = "raw_mean_zscore"
    trainer._feature_indices = None

    probs, actuals = trainer.predict_npy(str(tiny_npy), split="test")
    # Test split: 1 row per stock (val_end..end)
    # Row 0 (AAA): y[3] — target = 0.01 non-zero. valid.
    # Row 1 (BBB): y[7] — target = 0.01 non-zero. valid.
    # But wait — test rows are 3 and 7. Neither has target==0 in the test split.
    # Let me instead check val split where row 2 is the val row for AAA.

    # Reset for val test: row 2 in val split has target=0
    val_probs, _ = trainer.predict_npy(str(tiny_npy), split="val")
    # Val: AAA row=2 (target=0), BBB row=6 (target=0.01)
    # AAA val row must have NON-uniform prediction despite target==0
    # Since val returns 2 rows (1 per stock), first is AAA (target=0) — must not be uniform
    uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    assert not np.allclose(val_probs[0], uniform, atol=1e-6), (
        f"Row with target==0 got uniform prediction {val_probs[0]}; "
        "inference must predict all rows, not filter by target"
    )
