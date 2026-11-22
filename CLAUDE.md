# Starting a research session

When asked to run a new research iteration, read the [AutoResearch procedure](autoresearch/README.md) and the log of the previous iteration. The grounds for earlier decisions are in `autoresearch/records/` and `research/records/`.

- **Meeting topic:** state the observed result together with the question to decide this time. When using Alpha Squad, pass the previous result with `--context-file`. [Agenda example](alpha_squad/README.md)
- **Before the experiment:** write the baseline, the changed condition, the period, seeds, cost assumptions, and the deciding metric in the iteration log. When swapping a model, use the same data and evaluation settings, and record the YAML and the CLI overrides that were run. [JAX configuration](research/jax.md)
- **After the experiment:** record the measured values and exactly one verdict (`keep`, `discard`, or `crash`) in the results table. In the log, record the reason and the condition to change next. Write new iteration records in `autoresearch/runs/`.
- The synthetic-data demo verifies the training and prediction path. When recording market performance, use results from experiments run on the corresponding market data period.
