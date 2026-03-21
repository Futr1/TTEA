# Experiments

The repository includes experiment configs for the paper's main three task groups plus one optional extension:

- Web navigation
  - `configs/experiments/webarena.json`
  - `configs/experiments/miniwob.json`
- Collaborative software engineering
  - `configs/experiments/swebench_lite.json`
- Knowledge enhancement generation
  - `configs/experiments/pubhealth.json`
  - `configs/experiments/arc_challenge.json`
  - `configs/experiments/squad.json`
  - `configs/experiments/asqa.json`
- Optional translation extension
  - `configs/experiments/jrc_acquis.json`

The ablation settings are also wired into configuration files:

- `configs/experiments/ablation_top_level_objective.json`
- `configs/experiments/ablation_evolution.json`
- `configs/experiments/ablation_communication.json`

Paper-level result summaries are stored under `result/`:

- `result/web_navigation.json`
- `result/software_engineering.json`
- `result/knowledge_enhancement.json`
- `result/ablation.json`
- `result/translation.json`

Each experiment file contains:

- dataset binding
- metric list
- runtime budget
- paper target values
- persistence layout
- optional module switches for ablation
