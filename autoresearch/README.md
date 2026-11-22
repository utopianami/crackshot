# AutoResearch: using one result as the input to the next experiment

AutoResearch is the working method I set up so that a coding agent can continue research from where the previous experiment stopped. An agent session has no memory of the last one, so everything the next iteration needs is kept in files: **the experiment rules, a log per iteration, and a cumulative results table.** In the original project this workspace was a folder named `BRAIN`, which the agent read at the start of every session.

| File | Role |
| --- | --- |
| Protocol | The rules of the loop: what to read first, the order of steps, how to judge a result. [Original KOSPI protocol](records/kospi-protocol.md) |
| Iteration log | One dated file per session: observation, hypothesis, fixed conditions, result, reasoning, next question. [Example](records/binance-2026-04-28.md) |
| Results table | One row per experiment: date, dataset, model, trade count, performance, verdict, reason. [KOSPI table](records/kospi-results.csv) |

The original KOSPI protocol fixes the order: reproduce the previous best first (to tell a pipeline problem from a model problem), compare the plan against the actual config, then train and evaluate. It also requires every planned experiment to be run to a conclusion and recorded, including the ones that fail, and keeps a list of failed approaches and mistakes that must not be repeated.

## How one iteration runs

| Step | What to do | Evidence to leave |
| --- | --- | --- |
| 1. Pick the question | Choose one hypothesis to test, from an Alpha Squad proposal or the previous log. | Agenda, hypothesis, and what result would reject it |
| 2. Design the comparison | Fix the baseline's data period, seeds, costs, and metrics, and mark the one condition that changes. | Baseline config, changed config, exact run commands |
| 3. Implement and run | Select the JAX trainer and config, and compare the actual run values against the plan. | Run name, output path, evaluation period |
| 4. Judge | Add one row of measured numbers to the results table; write the interpretation and limits in the log. | `keep` (adopt), `discard` (reject), `crash` (run failed) |
| 5. Next question | Pass on any result that contradicted expectations, or any variable not yet controlled. | Follow-up hypothesis and a link to this iteration |

New iteration logs go in `autoresearch/runs/YYYY-MM-DD-topic.md`. When Alpha Squad opens its next meeting, it reads the most recent log as context. [Session start instructions](../CLAUDE.md) · [Meeting context loader](../alpha_squad/config.py)

A new research session can be started with an instruction like this:

> In the previous iteration the predictions of the two models were highly correlated. With the simple average as the baseline, test whether a weighted average after score standardization reduces maximum drawdown across the three evaluation periods. Keep the data period, costs, and seeds fixed, and record the grounds for adopting or rejecting it in the results table and the log.

## A case where the hypothesis changed

In the April 28 Binance research, the correlation between predictions was observed to vary by model combination. The hypothesis was that standardization and weighting would capture a diversification benefit, and ten combinations were compared. According to the record, the weighted combinations had a larger maximum drawdown than the simple average. The next iteration tested whether the model that led with three seeds kept the same rank with five. [Original log with observation, hypothesis, and verdict per cycle](records/binance-2026-04-28.md)

For KOSPI, each row of the table holds the date, dataset, model, trade count, performance, verdict, and reason. The April 2 rows include the decision to reject three models and carry the base-feature CS-MLP into follow-up experiments. [Results table](records/kospi-results.csv)

## Record index

These are the original files the loop produced, kept as written (in Korean). Internal dashboard links and a colleague's name were removed; nothing else was edited.

**KOSPI: 21 session logs, March 16 to April 15, 2026**

| Date | Session | Log |
| --- | --- | --- |
| 03-16 | 114 experiments on the first dataset | [log](records/kospi/2026-03-16.md) |
| 03-17 | Daily session: plan, backtest results, verdicts | [log](records/kospi/2026-03-17.md) |
| 03-18 | Daily session: plan, backtest results, verdicts | [log](records/kospi/2026-03-18.md) |
| 03-19 | Daily session: plan, backtest results, verdicts | [log](records/kospi/2026-03-19.md) |
| 03-20 | Extending from the top 200 to all KOSPI stocks (four files: first attempt, next-experiment plan, retry with 600 experiments, stocks outside the top 200 from a zero baseline) | [1](records/kospi/2026-03-20_1.md) · [2](records/kospi/2026-03-20_2.md) · [3](records/kospi/2026-03-20_3.md) · [4](records/kospi/2026-03-20_4.md) |
| 03-21 | Improving performance outside the top 200; loss and label changes plus model search | [1](records/kospi/2026-03-21.md) · [2](records/kospi/2026-03-21_2.md) |
| 03-25 | Improving KOSPI 200 accuracy | [log](records/kospi/2026-03-25.md) |
| 03-26 | Features from overnight global assets | [log](records/kospi/2026-03-26.md) |
| 03-27 | Sector ETF mapping and MASTER gating | [log](records/kospi/2026-03-27.md) |
| 03-28 | Per-stock beta times overnight signal | [log](records/kospi/2026-03-28.md) |
| 03-29 | New directions to exceed the reference model | [log](records/kospi/2026-03-29.md) |
| 03-30 | New best cumulative alpha | [log](records/kospi/2026-03-30.md) |
| 04-02 | New best under the new backtest standard | [log](records/kospi/2026-04-02.md) |
| 04-03 | Structural change based on an Alpha Squad meeting | [log](records/kospi/2026-04-03.md) |
| 04-07 | New best under the revised backtest settings | [log](records/kospi/2026-04-07.md) |
| 04-08 | Backtest redesign, training reproducibility, seed-tournament finding | [log](../research/records/kospi-2026-04-08.md) |
| 04-15 | Root cause of the TabM sentinel collapse | [log](records/kospi/2026-04-15.md) |

Alongside the logs: the [protocol](records/kospi-protocol.md), the [results table](records/kospi-results.csv), and the [don't-repeat list](records/kospi/dont-repeat.md) of failed approaches and mistakes.

**Binance: April 20 to April 28, 2026**

| Date | Record |
| --- | --- |
| 04-20 | [Research plan](records/binance/2026-04-20-plan.md), written from an Alpha Squad meeting: problem statement, phases, go/no-go gates |
| | [Phase 0 verdict](records/binance/phase0-verdict.md) on the entry gates |
| 04-22 | [Stage 4 final comparison](records/binance/2026-04-22-stage4-final-comparison.md): six scorer architectures across three windows |
| | [Stage 4 model review](records/binance/stage4-models-review.md): paper review and implementation checklist for added models |
| | [Seed variance summary](records/binance/stage4-seed-variance.md): three models, five seeds, three windows |
| | [v7 to v8 research log](records/binance/v7-v8-research-log.md), after a stride-anchor fix |
| 04-25 | [Session log](records/binance/2026-04-25-off30.md): running the offset-30 plan and diagnosing a fading 1-hour signal |
| 04-28 | [Session log](records/binance-2026-04-28.md): autonomous observe, hypothesize, experiment cycle on CS-MLP variants |

## What can be reproduced here

The original protocol and logs refer to the research server's paths and to a separate backtest engine. What runs directly from this repository is training on synthetic data and the tests. Repeating an iteration on real market data requires preparing the data period and the evaluation environment, then following the steps above. [Training commands and demo](../research/jax.md)
