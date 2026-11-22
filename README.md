# Crackshot

A research environment for cross-sectional return prediction in financial markets, written in JAX and designed to be operated together with an AI coding agent.

## What this project is

The research question is the same in both markets studied here: at each point in time, rank a universe of assets by expected short-horizon return, and check whether that ranking still earns money after trading costs.

- **KOSPI stocks:** predict the return bucket (five classes) 120 minutes ahead for each stock.
- **Binance USDT-M perpetual futures:** build a long/short portfolio across about 300 assets, rebalanced every four hours, on returns with the BTC market direction removed.

In this kind of research the signal is weak, most ideas fail, and a result can change with the seed or with the backtest settings. Progress depends less on any single model than on how many ideas can be tested fairly and on whether what was learned, including the failures, is still available for the next experiment. So the project is built around the research process itself. A model is one replaceable part of it.

## Key strengths

- **Fast, fair model comparison.** A new model is one architecture file, one trainer, and a YAML entry. The shared training code does not change, so two models are compared on identical data, labels, and evaluation.
- **The same code across markets.** Dataset-specific values live in config, not in model code. The same trainer runs on KOSPI stocks and on Binance futures.
- **Research that continues across sessions.** Every experiment leaves a hypothesis, the fixed conditions, the numbers, and a verdict in files. A new session, human or agent, starts from those files without relying on anyone's memory.
- **Research questions chosen by structured debate.** A multi-agent meeting challenges each proposal from the feature, evaluation, and implementation sides, using related papers and past results, before any compute is spent.
- **Honest evaluation.** Negative results are recorded with their reasons. The case studies below include a collapsed model, an ensemble that did not help, and a measurement of how much the backtest settings alone move the result.
- **Checkable without market data.** 35 tests and a synthetic-data demo run from a fresh clone with no external account.

## How it is organized

The strengths above come from three parts that feed each other:

| Part | Role | Guide |
| --- | --- | --- |
| **JAX experiment framework** | Runs the experiment. Models and inputs are swappable. | [research/jax.md](research/jax.md) |
| **AutoResearch** | Keeps the research loop going. Records each iteration so the next one builds on it. | [autoresearch/README.md](autoresearch/README.md) |
| **Alpha Squad** | Decides what to test next. Turns an observation into a research question and an experiment spec. | [alpha_squad/README.md](alpha_squad/README.md) |

```
Alpha Squad            AutoResearch                 JAX framework
(what to test next) -> (hypothesis, baseline,    -> (swap model or input,
        ^               fixed conditions)            train, evaluate)
        |                                                  |
        +------ verdict and reasons, written to files <----+
```

### 1. JAX framework: swap the model, swap the input

**Swap the model.** TabM and FT-Transformer differ in architecture and in how they train, yet both run from the same training script on the same data and labels. The forward pass lives in an Equinox module; optimization and prediction live in a trainer. Each trainer registers its own name and default config, and the entry point picks the implementation from `model.name` in the YAML. Adding a model does not touch the shared training script. [Registry](src/training/registry.py) · [Training entry point](scripts/train.py) · [TabM config](configs/tab_m_binance_phase1.yaml) · [FT-Transformer config](configs/ft_transformer_binance_phase1.yaml)

**Swap the input.** Everything that depends on the dataset (data path, which label columns are the horizons, the return-bucket edges) sits in the `data` and `dataset` blocks of the YAML. The same CS-MLP trainer runs on both markets by changing those blocks only. [KOSPI config](configs/cs_mlp_200_base.yaml) · [Binance config](configs/cs_mlp_binance_phase1.yaml)

**Try it.** From the repository root, this trains both models for one epoch on a fixed synthetic dataset. Each trainer returns 5-class probabilities for the same 16 test samples. Requires Python 3.12 and `uv`.

```bash
uv run --locked python -m examples.compare_trainers
```

### 2. AutoResearch: from one result to the next experiment

Each iteration writes down the hypothesis, the fixed conditions, and the evaluation method, then checks that the implemented config matches the plan before running. Afterwards the numbers and the verdict (`keep`, `discard`, or `crash`) go into a cumulative results table, and the reasoning goes into a dated research log. The next session starts by reading those files.

For example, on April 28 a weighted ensemble was tested after looking at the correlation between model predictions. It failed to beat the simple average, so the next iteration checked how stable the ranking was across more seeds. [Procedure and worked example](autoresearch/README.md) · [Research log for that day](autoresearch/records/binance-2026-04-28.md)

### 3. Alpha Squad: setting the next research question

The researcher submits an observation and a question as the agenda. Five agents with different roles challenge it using related papers and past experiments, then propose a baseline to compare against and follow-up experiments. On April 20 the agenda was: **"Once costs are included, the long/short performance of all three classification models collapses. How should the training objective change?"** The meeting proposed a staged validation plan; the actual research kept the existing data and tested a portfolio loss first. [Agenda input and meeting flow](alpha_squad/README.md) · [Meeting transcript](alpha_squad/meetings/meeting_20260420_171228.md)

## Case studies

| Case | What was found | Details |
| --- | --- | --- |
| Binance portfolio loss | Why every asset received the same score under the first config, and a period where combining three seeds still did not overcome trading costs | [Design, results, limits](research/binance.md) |
| KOSPI evaluation reproducibility | After fixing a bug in the inference filter, evaluation variance was measured with three same-seed trainings across eight backtest conditions | [Experiment and regression test](research/kospi.md) |

## Verification

```bash
uv run --locked python -m pytest -q
```

The tests cover trainer auto-registration, config overrides, the portfolio training step, and the inference filter.

The numbers in the case studies come from the records written at the time. Re-running full training and backtests needs the market data and preprocessed arrays for those periods, which are not in this repository.

## Repository layout

```
src/            JAX framework: architectures, trainers, registry, losses, config schema
configs/        Experiment configs (model, data, and dataset blocks)
scripts/        Training entry point, evaluation, Binance data builders and backtest
examples/       Synthetic-data demo of swapping trainers
autoresearch/   Research loop: procedure, original protocol, logs, results table
alpha_squad/    Multi-agent research meeting: agents, literature pipeline, one transcript
research/       Write-ups of the two case studies, with the original records
tests/          Regression and unit tests
```

## Scope and language

This is a curated subset of a larger research codebase. It does not include raw market data, trained weights, model-serving code, or live-trading code. Some records mention an external backtest engine and server paths that are not part of this repository.

The guides and write-ups are in English. The original records (research logs, the meeting transcript, the results table) are kept as written, in Korean, each with a short English note at the top. Some code comments and commit messages are also in Korean.
