# Alpha Squad

A research-meeting tool for deciding which financial ML experiment is worth the time. It takes past experiments and related literature as input, puts a proposal through role-based objections, and produces an experiment plan with a comparison target, the quantities to measure, and the conditions for rejecting it. Its output is the starting point of an [AutoResearch](../autoresearch/README.md) iteration.

## Setting the topic

The researcher passes the question to decide as the first argument of `meet`. `--context-file` carries observations and constraints from the previous iteration. The same topic is handed to the literature search, the role-based discussion, and the meeting record. The agenda below, for example, makes a failure (score variance collapsing to zero) the starting point of the next meeting.

```bash
uv run --locked --extra alpha-squad python -m alpha_squad.run meet \
  "The portfolio scores converged to a constant. Which penalty should be removed first, and what is the control?" \
  --context-file research/binance.md --rounds 2
```

[Topic CLI](run.py) · [Code that passes the topic and saves the meeting](meeting/orchestrator.py)

## What was actually adopted from the April 20 meeting

| Stage | Binance long/short case |
| --- | --- |
| Problem brought to the meeting | With trading costs included, the baseline long/short Sharpe of all three classification models turned negative. Training the portfolio directly was already on the agenda. |
| Review by the agents | The feature specialist examined the beta estimate and data leakage; the evaluation specialist examined the target definition. The transcript contains a staged proposal, from validating the residual target to comparing portfolio losses. |
| The researcher's choice | To check feasibility first, the existing data arrays and 4-hour rebalancing were kept, and the portfolio-loss experiment was moved forward. |
| Answer from the experiment | Under the first config the per-asset scores converged to a constant. A later config restored the score differences, and it also showed a condition under which several seeds fail together in the same period. |

[Meeting transcript](meetings/meeting_20260420_171228.md) · [Adopted configs and results](../research/binance.md)

## Meeting index

All 16 transcripts are in [`meetings/`](meetings/), kept as generated (in Korean). Each one starts with the topic the researcher submitted.

| Date | Topic | Transcript | Related research log |
| --- | --- | --- | --- |
| 03-20 | Next direction after the TabM ensemble: new architecture, better features, or redefining the problem | [14:28](meetings/meeting_20260320_142853.md) | |
| 03-20 | Strategy for stocks outside the top 200; produced the 03-21 research plan | [16:02](meetings/meeting_20260320_160208.md) | [03-21](../autoresearch/records/kospi/2026-03-21.md) |
| 03-21 | Training on an alpha label; self-supervised pretraining on raw candles | [12:16](meetings/meeting_20260321_121603.md) | [03-21 (2)](../autoresearch/records/kospi/2026-03-21_2.md) |
| 03-21 | Searching for a fundamentally different model architecture | [12:35](meetings/meeting_20260321_123551.md) | [03-21 (2)](../autoresearch/records/kospi/2026-03-21_2.md) |
| 03-25 | Improving KOSPI 200 accuracy (three sessions, ending on representation learning from raw candles) | [02:08](meetings/meeting_20260325_020839.md) · [03:55](meetings/meeting_20260325_035528.md) · [04:47](meetings/meeting_20260325_044727.md) | [03-25](../autoresearch/records/kospi/2026-03-25.md) |
| 03-26 | Redefining the problem: prioritizing 14 identified issues | [05:41](meetings/meeting_20260326_054131.md) | [03-26](../autoresearch/records/kospi/2026-03-26.md) |
| 03-26 | How to change the data and problem definition once tuning has hit its limit | [07:06](meetings/meeting_20260326_070609.md) | [03-26](../autoresearch/records/kospi/2026-03-26.md) |
| 03-26 | Lowest IC is in the first 90 minutes of the session: which overnight global-asset data to add | [07:39](meetings/meeting_20260326_073907.md) | [03-26](../autoresearch/records/kospi/2026-03-26.md) |
| 03-26 | Designing overnight features that capture path quality as well as return size | [08:09](meetings/meeting_20260326_080937.md) | [03-26](../autoresearch/records/kospi/2026-03-26.md) |
| 04-02 | Hyperparameter tuning has stalled; which structural change comes next | [15:28](meetings/meeting_20260402_152856.md) | [04-03](../autoresearch/records/kospi/2026-04-03.md) |
| 04-07 | Experiment plan under the revised backtest | [09:55](meetings/meeting_20260407_095502.md) | [04-07](../autoresearch/records/kospi/2026-04-07.md) |
| 04-20 | Problem definition for the Binance long/short alpha strategy | [07:39](meetings/meeting_20260420_073920.md) | [Binance plan](../autoresearch/records/binance/2026-04-20-plan.md) |
| 04-20 | Review of the Phase 1 baselines and the portfolio-loss trainer design | [17:12](meetings/meeting_20260420_171228.md) | [Binance write-up](../research/binance.md) |
| 04-23 | Decomposing where the Binance portfolio's profit actually comes from | [10:36](meetings/meeting_20260423_103611.md) | |

## Implementation

1. **Collecting literature candidates.** A hypothetical abstract is generated from the research question and used to search a local paper index; the hits are merged with a list recommended by a language model. After de-duplication and ranking by relevance, the result becomes the meeting context. [Literature pipeline](knowledge/researcher2_pipeline.py)
2. **Role-based critique.** The architecture specialist proposes; the feature, evaluation, and implementation specialists review in turn. The original proposer answers the objections, and the experiment PM writes up the spec. [Speaking order and meeting flow](meeting/orchestrator.py) · [Role prompts](agents/)
3. **Records.** Each meeting is saved as a Markdown transcript. The knowledge graph and vector memory can be reused in later searches. [LightRAG integration](knowledge/lightrag_backend.py) · [ChromaDB memory](memory/store.py)

## Running it

The representative meeting can be read directly in the transcript above. To run a new meeting, use the command below from the repository root. It needs a logged-in Claude CLI and downloads an embedding model on first use. Local paper search needs an abstract index prepared separately. [Index collection](knowledge/collect_papers.py)

```bash
uv run --locked --extra alpha-squad python -m alpha_squad.run meet --help
```

`meet` calls an external model. `--model` and `--pm-model` select the meeting models. The core research tests run without any account. [How to run the tests](../README.md)
