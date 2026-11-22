# Binance: what happens when the training objective becomes portfolio return?

## Starting point

When three kinds of classification models predicted up/down moves and picked assets, the baselines brought to the meeting all showed negative long/short Sharpe under a 5 bp one-way cost assumption. To narrow the gap between prediction ranking and after-trade performance, I designed an experiment in which the model outputs a score per asset, the scores are turned into long/short weights, and training is done directly on residual returns. [Baselines and meeting agenda](../alpha_squad/meetings/meeting_20260420_171228.md)

## Design and the first failure

298 assets were compared every four hours. Market-direction exposure was adjusted with each asset's beta to BTC, and a differentiable top-k operator produced weights corresponding to 10 longs and 10 shorts. The first config penalized beta exposure and turnover. Validation loss converged to zero, but the standard deviation of per-asset scores in the saved model was also zero. If every asset gets the same score, longs and shorts cancel out, so the loss and the positions have to be read together. [First config](../configs/cs_mlp_portfolio_binance.yaml) · [Asset selection](../src/training/heads/long_short_topk.py) · [Loss](../src/training/losses/portfolio.py)

| Measure | First config | Follow-up config |
| --- | ---: | ---: |
| Std of per-asset scores | 0.0 | 4.44 |
| Pearson correlation, validation | 0.011 | 0.0214 |
| Pearson correlation, test | Undefined (constant prediction) | 0.0164 |

The follow-up config set the beta and turnover penalties and weight decay to zero, and also changed feature standardization, dropout, and other settings. It confirmed that the scores separate assets again. This comparison alone cannot identify which change caused the recovery. [Follow-up config](../configs/cs_mlp_portfolio_binance_v2.yaml) · [Experiment record](records/binance-portfolio.md)

## A period that ensembling did not fix

The later v7 model was evaluated on March 1 to April 30, 2025 (W2). Even after combining the positions of three training seeds, turnover was high and the Sharpe after costs was negative.

| W2 evaluation | Seed 42 | Seed 123 | Seed 2026 | Three-seed ensemble |
| --- | ---: | ---: | ---: | ---: |
| Test net Sharpe | 0.04 | −0.06 | −1.12 | −0.32 |
| Turnover (units as recorded) | 35.2 | 35.3 | 33.3 | 34.0 |

In this period, changing the seed was not enough to cancel a trading pattern the seeds had in common. Three seeds in one period are also not enough to estimate the failure probability in other periods. The numbers are backtest results using the residual returns and cost model of the time; confirming actual fills and funding cash flows would need separate data. [W2 config](../configs/cs_mlp_portfolio_binance_v7_w2.yaml) · [Per-seed measurements](records/binance-portfolio.md) · [Backtest](../scripts/binance/backtest_portfolio.py)

## How to verify

`uv run --locked python -m pytest -q tests/binance` checks the features, asset selection, loss, and training step without external market data. With the training data of the time, the entry point can be run as `uv run --locked python -m scripts.train -c configs/cs_mlp_portfolio_binance_v2.yaml data.data_path=/path/to/npy`. [Array layout](../scripts/binance/build_npy_phase2.py)
