# JAX experiment structure

To change only the model on the same data, or only the data under the same model, three kinds of choices have to be kept apart: the model architecture, the training method, and the dataset. In this repository Equinox handles the forward pass, trainers built on JAX and Optax handle training, and OmegaConf configs hold the per-experiment choices.

| What you want to change | Where to change it | Example |
| --- | --- | --- |
| How features become probabilities | The model's Equinox module | [TabM](../src/models/architectures/tab_m.py) · [FT-Transformer](../src/models/architectures/ft_transformer.py) |
| Loss, optimization, inference | The model's trainer, which implements `BaseTrainer` | [TabM trainer](../src/training/trainers/tab_m.py) · [FT-Transformer trainer](../src/training/trainers/ft_transformer.py) |
| Model defaults and experiment conditions | Defaults in a dataclass; per-experiment changes in YAML and on the CLI | [TabM defaults](../src/config/models/tab_m.py) · [Experiment config](../configs/tab_m_binance_phase1.yaml) |
| Which data, horizons, and return buckets | The `data` and `dataset` blocks of the YAML | [KOSPI](../configs/cs_mlp_200_base.yaml) · [Binance](../configs/cs_mlp_binance_phase1.yaml) |

## Swapping the model

Steps to plug in a new model:

1. Create the forward-pass module in `src/models/architectures/`. Decide the random keys it needs during training and inference, and its output shape.
2. Define the model config in `src/config/models/`. In `src/training/trainers/`, implement `fit_from_npy`, `predict_npy`, `save`, and `load`, and pass the model name and default config to `@register_trainer`.
3. Put the registered name in `model.name` in the YAML. At run time `discover_trainers()` imports the trainer modules and `load_config()` merges the model defaults, the YAML, and CLI overrides. [Common interface and auto-registration](../src/training/registry.py) · [Config merging](../src/config/__init__.py)

No existing file needs to change: not the root config, not `__init__.py`, not the training script.

The two configs below use the same split, return buckets, and evaluation horizon. Point them at a prepared Phase 1 NPY directory and they run through the same entry point.

```bash
uv run --locked python -m scripts.train -c configs/tab_m_binance_phase1.yaml data.data_path=/path/to/npy
uv run --locked python -m scripts.train -c configs/ft_transformer_binance_phase1.yaml data.data_path=/path/to/npy
```

## Swapping the input

Dataset-dependent values never appear in model or trainer code. They are read from two config blocks:

- `data`: where the arrays are and how much to sample (`data_path`, `sample_ratio`, split dates, symbol filters).
- `dataset`: how to read the labels (`horizon_indices` for which label columns are the prediction horizons, `primary_horizon`, `vol_idx`, and `class_bins` for the return-bucket edges).

The same CS-MLP trainer was used on both markets by changing only these blocks:

| | KOSPI ([config](../configs/cs_mlp_200_base.yaml)) | Binance ([config](../configs/cs_mlp_binance_phase1.yaml)) |
| --- | --- | --- |
| `data.data_path` | `data/kospi/260401_200_base/` | `data/binance/npy_phase1/` |
| `dataset.horizon_indices` | `[4, 5, 6]` | `[0, 1, 2]` |
| `dataset.class_bins` | `[-0.01, -0.003, 0.003, 0.01]` | `[-0.02, -0.005, 0.005, 0.02]` |
| `model.name` | `cs_mlp` | `cs_mlp` |

Any of these can also be overridden on the command line, as `data.data_path=...` is above. [Config schema](../src/config/common.py)

## Checking the swap without market data

```bash
uv run --locked python -m examples.compare_trainers
```

This example feeds the same synthetic samples to two models and, after training, checks the prediction shape and that probabilities sum to one. [Example](../examples/compare_trainers.py) · [Tests for auto-registration and config precedence](../tests/strict/test_trainer_discovery.py)
