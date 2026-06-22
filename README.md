# Batch Size in LLM-Based CrowdRE Sector Classification

Replication package for the paper:

> **How Many Requirements Should an LLM Classify at Once? A Cost and Accuracy Study of Batch Size for CrowdRE Sector Classification**

[![DOI](https://zenodo.org/badge/1270908885.svg)](https://doi.org/10.5281/zenodo.20713661)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the code, datasets, and aggregated results needed to reproduce the study. The full raw artifacts (per-batch model responses, detailed JSON logs, and run checkpoints) are large and are archived separately on Zenodo (see [Large Artifacts](#large-artifacts)).

---

## Overview

We study how the **batch size** (the number of requirements packed into a single LLM inference call) affects classification accuracy, token cost, and runtime for two CrowdRE sector classification tasks:

- **Quaternary** (4 classes: Energy, Entertainment, Health, Safety)
- **Quinary** (5 classes: the above plus the ambiguous "Other")

Five open-source model configurations are evaluated across seven batch sizes (1 to 64) and two prompting strategies (Enhanced Zero-Shot and Few-Shot with Reasoning), and the results are combined into a three-objective (accuracy, tokens, latency) Pareto analysis.

| Model | Class | Architecture |
|---|---|---|
| Gemma-4-31B | Tiny | Dense |
| Mixtral-8x22B | Small | MoE |
| Nemotron-Ultra-253B | Medium | Dense |
| Llama-4-Maverick | Large | MoE |
| DeepSeek-V3.2 | Ultra | MoE |

---

## Repository Structure

```
llm-batch-size-crowdre/
├── README.md
├── requirements.txt
├── LICENSE
├── scripts/
│   ├── run_experiment.py   # one unified runner for every model and task
│   ├── run_all.sh          # guide script: reproduce the whole study end to end
│   └── analysis/           # aggregation, tables, and figures
│       ├── analyze.py
│       ├── plot_u_shape_and_pareto.py
│       ├── plot_failure_rate.py
│       ├── plot_think_comparison.py
│       ├── pareto_all_batches_4_5.py
│       ├── make_dualaxis_fig.py
│       ├── quinary_confusion.py
│       └── timecheck.py
├── data/                   # the input requirement datasets, one CSV per task
│   ├── Quaternary/
│   ├── Quinary_Batch/
│   └── ...
├── results/                # aggregated outcomes per model and task (CSV)
│   └── <model>/<task>/
│       ├── batch_summary_aggregated.csv
│       └── batch_results.csv
└── figures/                # key figures from the paper
```

- `batch_summary_aggregated.csv` holds, for each batch size and prompt, the Macro F1, precision, recall, accuracy, tokens per requirement, failure rate, and execution time.
- `batch_results.csv` holds the per-run records used to build the aggregates.
- `results/Quinary_confusion_pooled.csv` holds the pooled Quinary confusion matrix (true vs predicted class counts per batch size, summed over the five default models and both prompts). It backs the "Other"-sink numbers in the paper's RQ2 (predicted-Other counts, "Other" precision and recall, and the share of errors assigned to "Other"); regenerate it with `quinary_confusion.py`.

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/anonymousscot-jpg/llm-batch-size-crowdre.git
cd llm-batch-size-crowdre
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducing the Results

### 1. Re-run the experiments (optional, requires API access)

A single unified runner, `scripts/run_experiment.py`, handles every model and task. **No keys are hardcoded**; supply your own through environment variables:

```bash
# NVIDIA-hosted models (Llama-4-Maverick, Nemotron-Ultra-253B):
export NVIDIA_API_KEYS="key1,key2"          # one or more keys, comma-separated (round-robin)

# OpenAI-compatible / Ollama endpoints (DeepSeek-V3.2, Gemma-4-31B, Mixtral-8x22B):
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"              # or your endpoint key
```

List the available configurations, then run one:

```bash
cd scripts
python run_experiment.py --list
python run_experiment.py --model gemma-4-31b --task Quinary --dry-run   # quick smoke test (batch size 1)
python run_experiment.py --model llama-4-maverick --task Quaternary
```

To reproduce the **entire study** (all models, both target tasks), use the guide script:

```bash
cd scripts
bash run_all.sh                # the two target tasks
```

The runner sweeps batch sizes `1 2 4 8 16 32 64` and both prompts, validates every response (valid JSON, exact item count, exact IDs, up to three retries), and writes results into `results/<model>/<task>/`. Inference settings are fixed (temperature 1.0, top-p 0.95) and the output token budget scales with the batch size as `max_tokens = 4096 + 180 * batch_size`. Useful flags: `--prompts`, `--batch-sizes`, `--reps`, `--out`, `--dry-run`.

### 2. Regenerate the analysis and figures (no API access needed)

The analysis scripts read only the CSVs already provided in `results/`:

```bash
python scripts/analysis/analyze.py                 # heatmaps, per-task plots, summary tables
python scripts/analysis/plot_u_shape_and_pareto.py # U-shape, 3D Pareto
python scripts/analysis/plot_failure_rate.py       # structural failure rate by prompt (matplotlib)
python scripts/analysis/plot_think_comparison.py   # Think ON vs OFF, F1 vs batch size (matplotlib)
python scripts/analysis/pareto_all_batches_4_5.py  # 3D efficiency-score Pareto table (LaTeX)
python scripts/analysis/make_dualaxis_fig.py       # dual-axis F1 vs token-cost trends
python scripts/analysis/quinary_confusion.py       # pooled Quinary confusion matrix + RQ2 Other-sink stats (needs detailed logs)
```

---

## Datasets

The requirements are **not synthetic**. They are real, crowd-generated software requirements sampled from a publicly available CrowdRE dataset of 2,966 smart-home requirements collected in a prior study. For each task we sample a class-balanced subset whose size divides evenly across every batch size up to 64 (80 requirements per class for Quaternary, 64 per class for Quinary).

---

## Large Artifacts

The complete raw outputs are too large for this repository and are archived on Zenodo with a citable DOI:

- Per-batch raw model responses and detailed JSON logs
- Run checkpoints
- The full set of generated figures and exported reports

**Zenodo archive:** https://doi.org/10.5281/zenodo.20713661

---

## Citation

If you use this artifact, please cite the paper:

```bibtex
@inproceedings{batchsize2027,
  title     = {How Many Requirements Should an LLM Classify at Once? A Cost and Accuracy Study of Batch Size for CrowdRE Sector Classification},
  booktitle = {Proceedings of the 18th Innovations in Software Engineering Conference (ISEC)},
  year      = {2027}
}
```

and the Zenodo archive (DOI above).

## License

Code is released under the [MIT License](LICENSE). The datasets are redistributed under the terms of their original source; see the dataset's original publication for details.
