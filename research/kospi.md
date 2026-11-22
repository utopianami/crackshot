# KOSPI: why did results vary between models trained with the same config?

While repeatedly training a CS-MLP that predicts the 120-minute-ahead return bucket for KOSPI stocks, backtest results differed widely even between runs of the same config. Before changing the model, I checked whether prediction rows were being dropped, then put the same three models through several evaluation conditions to measure where the difference grows. [Training config](../configs/cs_mlp_200_base.yaml) · [April 8 research log](records/kospi-2026-04-08.md)

## Why rows disappeared at inference

Training excluded rows whose target return was zero. The same condition was also being applied at inference, so some rows were filled with uniform probabilities over the five classes. The target return is not known at prediction time. The inference path was changed so the model predicts rows with a zero target too, and a small synthetic dataset containing such rows now checks the behavior. [Trainer](../src/training/trainers/cs_mlp.py) · [Regression test](../tests/test_cs_mlp_inference_filter.py)

## Under which conditions should the same predictions be evaluated?

With data and seed fixed and the GPU determinism option off, the model was trained three times, and each model was run through the same eight backtest conditions. The results below are for data whose test period ends on April 4, 2026. Units are KRW; the standard deviation is over the cumulative alpha PnL of the three runs.

| Condition | Evaluation settings | Mean cumulative alpha | Std (approx.) |
| --- | --- | ---: | ---: |
| C6 | Per-stock cap 2.5M KRW, weighted score, fixed slippage | −4,898,851 | 30,000 |
| OLD | Per-stock cap 2.5M KRW, base score, fixed slippage | −2,913,265 | 81,000 |
| C5 | Per-stock cap 7.5M KRW, base score, IC-based order sizing and dynamic slippage | +5,036,017 | 304,000 |
| NEW | Same execution settings as C5, weighted score | +133,932 | 807,000 |

C5 and NEW differ only in the weights that turn class probabilities into a score. Under NEW, one of the three runs had negative cumulative alpha. Judging model improvement from a single positive result under this condition would attribute to the model a difference that the score transform and the backtest created. The small standard deviation of C6 should likewise be read as a consistent loss. [Per-condition results for all eight conditions and cost assumptions](records/kospi-sensitivity.md)

There were three repetitions. In a follow-up comparison that retrained four times on a different data period, the ordering of the per-condition standard deviations also changed. The conclusion these records support is that model comparisons need the data period, the seed-selection method, and the evaluation condition fixed together, with repeated measurement. [Follow-up comparison](records/kospi-sensitivity.md)

`uv run --locked python -m pytest -q tests/test_cs_mlp_inference_filter.py` verifies inference on zero-target rows. The table summarizes records produced at the time by an external backtest engine.
