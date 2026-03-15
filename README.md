# TTEA: A Self-Reinforcing Multi-Agent Approach for Large-Scale Complex Systems

<div align="center">
  
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-f37b21?style=flat&logo=huggingface&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>


## Abstract

With the rapid development of large-scale models, intelligent agents have become increasingly capable of handling complex tasks, while multi-agent systems have also grown more complex. Enabling agents with different goals and architectures to collaborate effectively in a shared environment is therefore crucial. Existing work has largely focused on agent performance in isolated scenarios, while paying less attention to system-level enhancement. To address this gap, we propose **Top-level Target Entity Association (TTEA)**, a self-reinforcing multi-agent framework that improves system-wide performance through optimized architecture and efficient collaboration. TTEA combines four key ingredients: top-level objective guidance, an agent evolution mechanism, cross-level knowledge collaboration, and dynamic task dispatching. Specifically, the top-level objective provides a global perspective on agent behavior through a global utility function and a system impact assessment module; the evolution and communication mechanism uses skill reinforcement, learning, and elimination operators to drive self-evolution and improve resource utilization; cross-level knowledge collaboration enables information aggregation and sharing through category leaders and a global leader; and dynamic task dispatch monitors and adjusts task allocation in real time. Experiments on public benchmarks for web navigation, translation, and knowledge-enhanced generation show strong generalization ability and stable performance.

## Framework Overview

<p align="center">
  <img src="docs/ttea_figure4.png" alt="Full workflow of the TTEA framework" width="95%">
</p>
<p align="center">
   End-to-end workflow of the TTEA framework.
</p>

## Highlights

- Unified implementation of top-level objective optimization, element evolution, cross-level knowledge synergy, and dynamic task dispatch.
- Reproduction configs for **7 datasets** across **3 task groups**: WebArena, MiniWoB++, JRC-Acquis, PubHealth, ARC-Challenge, SQuAD, and ASQA.
- Real execution paths for both text tasks and browser-based web tasks.
- Config-driven training, inference, evaluation, checkpointing, prediction export, and trace persistence.
- Built-in ablation configs for removing top-level objective, evolution, or communication modules.



## Hardware and Environment

### Paper-reported hardware

According to the paper, model training and inference were conducted on:

- `1 x RTX 3090`
- `3 x NVIDIA A100`
- `2 x NVIDIA A6000`

### Recommended reproduction setup

- Full training reproduction: at least one GPU with `>= 240 GB` memory for seq2seq/classification experiments.
- Web navigation evaluation: Chromium environment plus sufficient CPU memory for browser tasks.
- Lightweight pipeline verification: CPU or a single consumer GPU is sufficient when using `--allow-placeholder`.

### Software requirements

- Python `>= 3.11`
- PyTorch `>= 2.3`
- Transformers `>= 4.41`
- Datasets `>= 2.19`
- Evaluate `>= 0.4`
- Optional integrations:
  - `playwright` for WebArena
  - `browsergym-core`, `browsergym-miniwob`, `gymnasium` for MiniWoB++
  - `mauve-text` for ASQA MAUVE evaluation

## Installation

### Base installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

### Optional dependencies

```bash
pip install -e .[dev]
pip install -e .[integration]
pip install -e .[web]
playwright install chromium
```

### Model checkpoint note

The default platform config uses:

- `model_name_or_path = google/flan-t5-base`
- `tokenizer_name_or_path = google/flan-t5-base`
- `local_files_only = true`

Therefore, you should either:

1. pre-download the model locally, or
2. change `local_files_only` to `false` in `configs/platform.json`.

## Datasets

Datasets are **not distributed** in this repository. Please download each dataset from its official source and place it in the corresponding directory under `data/datasets/`.

| Dataset | Local path | Expected files | Source |
| --- | --- | --- | --- |
| WebArena | `data/datasets/webarena` | `tasks.json`, `tasks.jsonl` | https://github.com/web-arena-x/webarena |
| MiniWoB++ | `data/datasets/miniwobpp` | `tasks.json`, `tasks.jsonl` | https://miniwob.farama.org/ |
| JRC-Acquis | `data/datasets/jrc_acquis` | `train/dev/test` bilingual files | https://opus.nlpl.eu/JRC-Acquis.php |
| PubHealth | `data/datasets/pubhealth` | `train.jsonl`, `dev.jsonl`, `test.jsonl` | https://huggingface.co/datasets/health_fact |
| ARC-Challenge | `data/datasets/arc_challenge` | `train.jsonl`, `dev.jsonl`, `test.jsonl` | https://allenai.org/data/arc |
| SQuAD | `data/datasets/squad` | `train.json`, `dev.json`, `test.json` | https://rajpurkar.github.io/SQuAD-explorer/ |
| ASQA | `data/datasets/asqa` | `train.json`, `dev.json`, `test.json` | https://github.com/google-research/language/tree/master/language/asqa |

You can inspect local dataset availability with:

```bash
ttea describe-datasets
```

If you prefer to run directly from source without installing the console script, use:

```bash
PYTHONPATH=src python -m ttea describe-datasets
```

## Quick Reproduction

After `pip install -e .`, the commands below can be executed through the `ttea` CLI. If needed, you can replace `ttea` with `PYTHONPATH=src python -m ttea`.

### 1. Inspect the experiment plan

```bash
ttea plan-experiment --experiment configs/experiments/jrc_acquis.json
```

### 2. Preview a small number of tasks

```bash
ttea dry-run --experiment configs/experiments/asqa.json --limit 2
```

### 3. Train text-task experiments

```bash
ttea train-experiment --experiment configs/experiments/jrc_acquis.json
ttea train-experiment --experiment configs/experiments/pubhealth.json
ttea train-experiment --experiment configs/experiments/arc_challenge.json
ttea train-experiment --experiment configs/experiments/squad.json
ttea train-experiment --experiment configs/experiments/asqa.json
```

### 4. Evaluate experiments

```bash
ttea run-experiment --experiment configs/experiments/jrc_acquis.json --split test
ttea run-experiment --experiment configs/experiments/webarena.json --split test
ttea run-experiment --experiment configs/experiments/miniwob.json --split test
```



## Main Reproduction Configurations

### Global platform parameters

The core system-level parameters are defined in `configs/platform.json`.

| Module | Key parameters |
| --- | --- |
| Top-level objective | `alpha=12.0`, `beta=8.0`, `delta=9.0`, `gamma=1.0`, `stability_floor=0.35`, `resource_budget=100.0` |
| Evolution | `skill_learning_rate=0.08`, `system_gain=1.75`, `skill_decay=0.015`, `elimination_threshold=-5.0`, `max_learning_attempts=3` |
| Communication | `encoder_dim=16`, `feature_grid_size=4`, `confidence_threshold=0.35`, `gaussian_smooth=true`, `fusion_mode=attention`, `fusion_heads=4` |
| Dispatch | `max_decomposition_depth=3`, `max_team_size=3`, `blocked_retry_limit=2`, `rebalance_interval=1` |
| Generation backend | `google/flan-t5-base`, `temperature=0.2`, `top_p=0.9`, `max_new_tokens=96` |

### Task-specific experiment parameters

| Experiment | Main settings |
| --- | --- |
| `webarena.json` | `max_steps=48`, `resource_budget=100`, `decomposition_depth=3`, live browser execution |
| `miniwob.json` | `max_steps=16`, `resource_budget=60`, `env_id=browsergym/miniwob.click-test-v0` |
| `jrc_acquis.json` | `batch_size=4`, `grad_acc=2`, `lr=3e-5`, `epochs=3.0` |
| `pubhealth.json` | `batch_size=8`, `lr=2e-5`, `epochs=2.0` |
| `arc_challenge.json` | `batch_size=8`, `lr=2e-5`, `epochs=2.0` |
| `squad.json` | `batch_size=8`, `lr=2e-5`, `epochs=2.0` |
| `asqa.json` | `batch_size=2`, `grad_acc=4`, `lr=2e-5`, `epochs=2.0` |

### Ablation configs

The following files reproduce the main ablation settings in the paper:

- `configs/experiments/ablation_top_level_objective.json`
- `configs/experiments/ablation_evolution.json`
- `configs/experiments/ablation_communication.json`



## Paper Target Results

The paper target values used by this repository are stored in `result/*.json` and mirrored in `configs/experiments/*.json`. Representative results include:

| Task group | Dataset | Paper target |
| --- | --- | --- |
| Web navigation | WebArena | `Avg SR = 41.7` |
| Web navigation | MiniWoB++ | `SR = 95.5` |
| Translation | JRC-Acquis | `En-De test BLEU = 62.19`, `De-En test BLEU = 67.01`, `En-Es test BLEU = 68.36`, `Es-En test BLEU = 69.67` |
| Knowledge enhancement | PubHealth | `Accuracy = 77.35` |
| Knowledge enhancement | ARC-Challenge | `Accuracy = 80.09` |
| Knowledge enhancement | ASQA | `Str-EM = 46.85`, `ROUGE-L = 46.21`, `MAUVE = 90.23` |
| Knowledge enhancement | SQuAD | `Exact Match = 29.14` |

## Output Artifacts

Each run is saved under `runs/` and can include:

- `config_snapshot.json`
- `plan.json`
- `summary.json`
- `metrics.json`
- `predictions.jsonl`
- `task_traces.jsonl`
- `training_summary.json`
- `training_history.jsonl`
- `checkpoints.json`



## Citation
```bibtex
@misc{Fu2026ttea,
  author = {Fu, Yuanshuang},
  title = {TTEA: A Self-Reinforcing Multi-Agent Approach for Large-Scale Complex Systems},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/Futr1/TTEA](https://github.com/Futr1/TTEA)}}
}

## License

This project is released under the [MIT License](LICENSE).
