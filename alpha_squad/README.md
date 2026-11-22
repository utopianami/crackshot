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
