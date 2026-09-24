# Reproducibility guide

## Running named experiments

Paper experiments are defined in `repro/experiments.yaml` and launched through
`repro/run.py`. The launcher resolves paths relative to the repository root,
sets `PYTHONPATH`, applies the requested seed and GPU, and prints the complete
command before execution.

```bash
python repro/run.py --list
python repro/run.py qwen-os-limbo --seed 42 --device 0 --dry-run
python repro/run.py qwen-os-limbo --seed 42 --device 0
```

## Run provenance

Record the following for each experiment:

- Git commit SHA
- named experiment ID
- complete command printed by the launcher
- random seed and task order
- model checkpoint or hosted-model identifier
- GPU type and visible-device mapping
- Docker image identifiers
- generated `config.yaml`, `metric.json`, and controller JSONL log

These fields capture the action space, budget semantics, task order, and model
runtime needed to interpret an online controller run.

## LIMBO controller configuration

The Qwen OS configuration uses retrieval memory size 64 with top-1 injection,
trimmed-trace rendering, `alpha=0.5`, and an adaptive cost coefficient initialized
at `0.12` with learning rate `0.10`. Its per-task target cost is `$0.000688`.
The controller selects among no-replay, recent/full-replay, and retrieved-replay
primitives with response-token caps from 256 to 1536 and task-round caps from 2
to 5.

Token limits selected by LIMBO apply to each model response. Round limits apply
to the complete task.

## Cost semantics

For local models, the cost tracker converts token counts through the repository
pricing table. This normalization supports controlled cost comparisons across
runs in the same codebase; it is not a cloud-inference invoice.

## Multi-seed aggregation

Use sample standard deviation (`n-1` denominator). For a single seed, report the
individual result and `n=1` rather than displaying a zero-width error bar.

```bash
python repro/summarize.py outputs/<seed42> outputs/<seed43> outputs/<seed44>
```

The summarizer reads each run's metrics, prints per-run accuracy and cost, and
reports their mean and sample standard deviation.
