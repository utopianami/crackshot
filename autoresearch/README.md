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

## What can be reproduced here

The original protocol and logs refer to the research server's paths and to a separate backtest engine. What runs directly from this repository is training on synthetic data and the tests. Repeating an iteration on real market data requires preparing the data period and the evaluation environment, then following the steps above. [Training commands and demo](../research/jax.md)
