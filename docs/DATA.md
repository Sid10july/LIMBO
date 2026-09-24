# Data setup

LIMBO uses the datasets and task environments distributed with
LifelongAgentBench. Dataset files are not mirrored in this repository.

## Download

The official dataset is hosted at:

https://huggingface.co/datasets/csyq/LifelongAgentBench

Download it into a temporary directory, inspect its top-level layout, and copy
the benchmark data into this repository's ignored `data/` directory. The task
YAMLs currently expect paths under:

```text
data/v0303/db_bench/processed/v0317_first500/
data/v0303/os_interaction/processed/v0409_tcc_9_to_12_first500/
```

Do not commit the resulting files. To confirm the resolved path before a full
run, use the original experiment with a small sample-limited copy of the task
config and inspect the first lines of its log.

## Runtime environments

DBBench starts MySQL containers dynamically. OS Interaction requires the
`local-os/default` image described in the root README. Docker must be available
to the process running the experiment or to the configured LAB task server.

## Hosted model credentials

GPT configurations read `OPENAI_API_KEY` from the process environment. The
OpenRouter variants read `OPENROUTER_API_KEY`. Never place either key directly
in a tracked YAML file.
