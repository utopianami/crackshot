"""Run two registered JAX trainers on the same small synthetic dataset."""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.config import load_config
from src.training.registry import create_trainer, discover_trainers


def write_sample(directory: Path) -> None:
    rng = np.random.default_rng(42)
    columns = ["x0", "x1", "x2", "x3"]
    metadata = {"feature_columns": columns, "n_features": len(columns)}
    (directory / "meta.json").write_text(json.dumps(metadata))
    for split, rows in [("train", 64), ("val", 16), ("test", 16)]:
        features = rng.standard_normal((rows, len(columns))).astype(np.float32)
        returns = 0.02 * features[:, :3]
        volume = np.ones((rows, 1), dtype=np.float32)
        np.save(directory / f"X_{split}.npy", features)
        np.save(directory / f"y_{split}.npy", np.column_stack([returns, volume]))


def main() -> None:
    discover_trainers()
    small_models = {
        "tab_m": ["num_members=2", "d_block=16", "n_blocks=1"],
        "ft_transformer": ["d_token=16", "n_heads=2", "n_layers=1"],
    }
    with tempfile.TemporaryDirectory() as temporary:
        data_dir = Path(temporary)
        write_sample(data_dir)
        for name, settings in small_models.items():
            overrides = ["epochs=1", "batch_size=16", "n_bins=4", *settings]
            config = load_config(
                f"configs/{name}_binance_phase1.yaml",
                [f"{name}.{item}" for item in overrides] + [f"data.data_path={data_dir}"],
            )
            trainer = create_trainer(config.model.name, cfg=config)
            trainer.fit_from_npy(data_dir)
            probabilities, actual = trainer.predict_npy(data_dir, split="test")
            assert probabilities.shape == (len(actual), 5)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            print(f"{name:16} {type(trainer).__name__:24} {probabilities.shape}")


if __name__ == "__main__":
    main()
