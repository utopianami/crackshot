# Crackshot

A Python (JAX) research project on cross-sectional return prediction in financial markets. This repository shows how I work through a research question: from the question, to data, a baseline, a model, validation, and an honest statement of limits, with the records of each step kept.

## The research question

At each point in time, rank a universe of assets by expected short-horizon return. Does that ranking still earn money after trading costs?

The question was studied in two markets:

- **KOSPI stocks:** predict the return bucket (five classes) 120 minutes ahead for each stock.
- **Binance USDT-M perpetual futures:** build a long/short portfolio across about 300 assets, rebalanced every four hours, on returns with the BTC market direction removed.

The domain is finance, but the difficulties are general ones for applied prediction research: the signal is weak, most ideas fail, and a result can change with the random seed or with the evaluation settings. So the project is built around the research process. A model is one replaceable part of it.

## How one research question was worked through

The Binance portfolio-loss study, step by step. Each row links to the code or the record for that step. Full write-up: [research/binance.md](research/binance.md).

| Step | What was done | Where to look |
| --- | --- | --- |
| **1. Question** | Three classification models had long/short Sharpe between −0.01 and +1.10 before costs, and all three turned negative at a 5 bp one-way cost. Question: does training directly on portfolio return close the gap between prediction ranking and after-cost performance? | [Meeting agenda and transcript](alpha_squad/meetings/meeting_20260420_171228.md) · [Research plan with go/no-go gates](autoresearch/records/binance/2026-04-20-plan.md) |
| **2. Data** | 5-minute bars sampled every four hours; 298 assets passed the universe filter. Targets are forward returns with the BTC component removed. Twelve candidate features were screened by information coefficient before any model was trained. | [BTC-residual features](src/features/binance_btc_residual.py) · [Funding features](src/features/binance_funding_clock.py) · [Array builder](scripts/binance/build_npy_phase2.py) · [Phase 0 verdict](autoresearch/records/binance/phase0-verdict.md) |
| **3. Baseline** | FT-Transformer, TabM, and CS-MLP trained as 5-class classifiers on the same split, labels, and evaluation, so the comparison is fair. | [Configs](configs/) · [Shared training entry point](scripts/train.py) |
| **4. Model** | The model outputs one score per asset; a differentiable top-k operator turns scores into long and short weights; the loss is the negative portfolio return on residual returns, with penalties on beta exposure and turnover. | [Loss](src/training/losses/portfolio.py) · [Long/short top-k head](src/training/heads/long_short_topk.py) · [Trainer](src/training/trainers/cs_mlp_portfolio.py) |
| **5. Validation** | The first config reached zero validation loss, and inspection showed why: every asset received the same score. Loss alone was not a sufficient check, so score dispersion was examined alongside it. Later configs were evaluated per seed and as a three-seed ensemble on a held-out window, after costs. | [Experiment record](research/records/binance-portfolio.md) · [Backtest](scripts/binance/backtest_portfolio.py) · [Unit tests](tests/binance/) |
| **6. Limits** | In the March to April 2025 window the three-seed ensemble still had negative net Sharpe (−0.32). The follow-up changed several settings at once, so the cause of the recovery is not identified. Three seeds in one window do not estimate failure risk elsewhere. | [Results and limits](research/binance.md) |
| **7. Next question** | Other iterations examined seed variance and ensemble weighting. | [Seed variance summary](autoresearch/records/binance/stage4-seed-variance.md) · [April 28 log](autoresearch/records/binance-2026-04-28.md) |

A second study asks a different kind of question: can the evaluation itself be trusted? Models trained from the same config gave different backtest results. The study found a bug that dropped rows at inference, fixed it with a regression test, then measured how much each of eight backtest conditions amplifies run-to-run variance. [research/kospi.md](research/kospi.md)

## What supports the process

Three parts make this repeatable. Each has its own guide.

| Part | Role in the process | Guide |
| --- | --- | --- |
| **JAX experiment framework** | Fair comparison. A new model is one architecture file, one trainer, and a YAML entry; shared training code does not change. Dataset-specific values live in config, so the same trainer runs on both markets. | [research/jax.md](research/jax.md) |
| **AutoResearch** | Continuity. Every iteration leaves a hypothesis, the fixed conditions, the numbers, and a verdict (`keep`, `discard`, `crash`) in files, so the next session starts from the record, not from memory. | [autoresearch/README.md](autoresearch/README.md) |
| **Alpha Squad** | Question selection. A multi-agent meeting challenges each proposal from the feature, evaluation, and implementation sides, using related papers and past results, before compute is spent. | [alpha_squad/README.md](alpha_squad/README.md) |

```
Alpha Squad            AutoResearch                 JAX framework
(what to test next) -> (hypothesis, baseline,    -> (swap model or input,
        ^               fixed conditions)            train, evaluate)
        |                                                  |
        +------ verdict and reasons, written to files <----+
```

## Evidence that the process ran

The records below are the original working files, not summaries written afterwards.

- **21 KOSPI session logs** from March 16 to April 15, 2026, each with the plan, the results, and the verdicts. [Index](autoresearch/README.md#record-index)
- **Binance records** from April 20 to April 28: the research plan, phase verdicts, model comparisons, seed-variance summaries, session logs. [Index](autoresearch/README.md#record-index)
- **16 Alpha Squad meeting transcripts**, most linked to the related research log. [Index](alpha_squad/README.md#meeting-index)
- **A don't-repeat list** of failed approaches and mistakes. [File](autoresearch/records/kospi/dont-repeat.md)
- **The git history**, which shows the framework, the models, and the records being built over the same weeks.

## Run it

Requires Python 3.12 and `uv`. No market data or external account is needed.

```bash
# 35 tests: trainer auto-registration, config overrides, portfolio loss and training step, inference filter
uv run --locked python -m pytest -q

# Train two different models for one epoch on the same synthetic data through the same entry point
uv run --locked python -m examples.compare_trainers
```

The numbers in the write-ups come from the records written at the time. Re-running full training and backtests needs the market data and preprocessed arrays for those periods, which are not in this repository.

## Repository layout

```
src/            JAX framework: architectures, trainers, registry, losses, config schema
configs/        Experiment configs (model, data, and dataset blocks)
scripts/        Training entry point, evaluation, Binance data builders and backtest
examples/       Synthetic-data demo of swapping trainers
research/       Write-ups of the two studies, with their original records
autoresearch/   Research loop: procedure, protocol, session logs, results table
alpha_squad/    Multi-agent research meeting: agents, literature pipeline, transcripts
tests/          Regression and unit tests
```

## Scope and language

This is a curated subset of a larger research codebase. It does not include raw market data, trained weights, model-serving code, or live-trading code. Some records mention an external backtest engine and server paths that are not part of this repository.

The guides and write-ups are in English. The original records are kept as written, in Korean, each with a short English note at the top. Some code comments and commit messages are also in Korean.
