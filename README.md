# LIMBO

**Lifelong Inference-Time Memory and Budget Optimization for LLM Agents**

LIMBO is an online controller for lifelong LLM agents. Before each task, it
selects a memory strategy and an inference budget, observes task correctness
and cost, and updates a lightweight contextual bandit without changing the
underlying language model.

This repository accompanies the ICTAI 2026 paper:

> Siddharth Sharma, Nilesh Prasad Pandey, Onat Gungor, and Tajana Rosing.
> *LIMBO: Lifelong Inference-Time Memory and Budget Optimization for LLM
> Agents.* 38th IEEE International Conference on Tools with Artificial
> Intelligence (ICTAI), 2026.

The code is built on
[LifelongAgentBench](https://github.com/caixd-220529/LifelongAgentBench) at
commit `d6f19b42eb358d9150379f0c68c2985c5a867520`.

## Artifact Contents

The repository provides the LIMBO controller, the DBBench and OS Interaction
experiment harness, named paper configurations, baseline implementations,
and result aggregation utilities. Model weights, benchmark data, API keys,
Docker state, and raw experiment outputs are not committed.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/controllers/dbbench_linucb.py` | LIMBO contextual-bandit controller and action spaces |
| `src/run_experiment.py` | Experiment integration and online update loop |
| `src/callbacks/instance/` | Replay, retrieval, compression, and baseline callbacks |
| `configs/assignments/experiments/` | Model- and environment-specific LAB configurations |
| `repro/experiments.yaml` | Named, reviewable experiment commands |
| `repro/run.py` | Safe launcher for named experiments |
| `repro/summarize.py` | Metric extraction and multi-seed aggregation |

## Requirements

- Linux with Python 3.11
- Docker for DBBench and OS Interaction environments
- A CUDA-capable GPU with enough memory for Qwen 2.5 7B or Llama 3.1 8B
- An OpenAI API key only for GPT-4o mini experiments
- Access to the Hugging Face model checkpoints used by the selected config

The experiments were developed on Linux. macOS is suitable for inspecting,
testing, and plotting the code, but not for reproducing the Docker/GPU runs.

## Setup

```bash
git clone https://github.com/Sid10july/LIMBO.git
cd LIMBO

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Install the optional baseline dependency only when needed:

```bash
pip install -r requirements-longllmlingua.txt  # LongLLMLingua runs
```

Download LifelongAgentBench data from the
[official dataset](https://huggingface.co/datasets/csyq/LifelongAgentBench)
and place it under `data/` with the layout expected by the YAML task configs.
The data directory is intentionally gitignored. See [`docs/DATA.md`](docs/DATA.md).

Build the task images:

```bash
docker pull mysql
docker pull ubuntu
docker build \
  -f scripts/dockerfile/os_interaction/default \
  scripts/dockerfile/os_interaction \
  --tag local-os/default
```

## Run a Paper Experiment

List the release commands:

```bash
python repro/run.py --list
```

Inspect a command without starting it:

```bash
python repro/run.py qwen-os-limbo --seed 42 --dry-run
```

Run the Qwen OS LIMBO configuration on GPU 0:

```bash
python repro/run.py qwen-os-limbo --seed 42 --device 0
```

Run all three matched seeds sequentially:

```bash
for seed in 42 43 44; do
  python repro/run.py qwen-os-limbo --seed "$seed" --device 0
done
```

The launcher changes no configuration files. It prints the complete command,
sets `PYTHONPATH` to the repository root, and invokes
`src/run_experiment.py`. Outputs are written under `outputs/<timestamp>/`.

## Core Qwen Experiments

| Method | DBBench | OS Interaction |
|---|---|---|
| No replay | `qwen-db-no-replay` | `qwen-os-no-replay` |
| Fixed replay | `qwen-db-fixed-k16` | `qwen-os-fixed-trim-k1` |
| Retrieval replay | `qwen-db-retrieval-k16` | `qwen-os-retrieval-k4` |
| LongLLMLingua | `qwen-db-longllmlingua-k16` | `qwen-os-longllmlingua-k1` |
| Group self-consistency | `qwen-db-gsc-k1` | `qwen-os-gsc-k1` |
| LIMBO | `qwen-db-limbo` | `qwen-os-limbo` |

## Hosted Models

For GPT-4o mini:

```bash
export OPENAI_API_KEY="..."
python repro/run.py gpt-db-no-replay --seed 42
```

Never place credentials in YAML files. `OPENAI_API_KEY` and
`OPENROUTER_API_KEY` should only be supplied through the environment.

## Results and Aggregation

Summarize one or more completed output directories:

```bash
python repro/summarize.py \
  outputs/2026-05-24-10-34-47 \
  outputs/2026-05-25-10-59-10 \
  outputs/2026-05-25-10-59-16
```

The script reports per-run accuracy and cost plus sample mean and sample
standard deviation. The release does not commit raw trajectories because they
may contain benchmark payloads and are substantially larger than the source.

## Reproducibility Notes

- Token limits selected by LIMBO are **per model response**, while round
  limits are **per task**.
- Qwen OS declares tool-call limits in its action profiles, but the current OS
  task implementation does not enforce them.
- Reported local-model dollar costs use the project pricing table to normalize
  token use; they are comparative estimates, not cloud invoices.
- Online experiments depend on task order. Always record the seed, config,
  exact commit, complete command, and generated `metric.json`.

More detail is available in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Citation

```bibtex
@inproceedings{sharma2026limbo,
  title     = {LIMBO: Lifelong Inference-Time Memory and Budget Optimization for LLM Agents},
  author    = {Sharma, Siddharth and Pandey, Nilesh Prasad and Gungor, Onat and Rosing, Tajana},
  booktitle = {Proceedings of the 38th IEEE International Conference on Tools with Artificial Intelligence},
  year      = {2026}
}
```

## Attribution and License

LIMBO is built on LifelongAgentBench. See [`NOTICE.md`](NOTICE.md) for upstream
attribution and licensing information.
