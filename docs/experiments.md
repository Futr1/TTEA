# Experiments

The repository includes experiment configs for the paper's three task groups and seven named datasets:

- Web navigation
  - [configs/experiments/webarena.json](d:\文件\论文\第二篇论文-etta多智能体\code\configs\experiments\webarena.json)
  - [configs/experiments/miniwob.json](d:\文件\论文\第二篇论文-etta多智能体\code\configs\experiments\miniwob.json)
- Translation
  - [configs/experiments/jrc_acquis.json](d:\文件\论文\第二篇论文-etta多智能体\code\configs\experiments\jrc_acquis.json)
- Knowledge enhancement generation
  - [configs/experiments/pubhealth.json](d:\文件\论文\第二篇论文-etta多智能体\code\configs\experiments\pubhealth.json)
  - [configs/experiments/arc_challenge.json](d:\文件\论文\第二篇论文-etta多智能体\code\configs\experiments\arc_challenge.json)
  - [configs/experiments/squad.json](d:\文件\论文\第二篇论文-etta多智能体\code\configs\experiments\squad.json)
  - [configs/experiments/asqa.json](d:\文件\论文\第二篇论文-etta多智能体\code\configs\experiments\asqa.json)

The ablation settings are also wired into configuration files:

- [configs/experiments/ablation_top_level_objective.json](d:\文件\论文\第二篇论文-etta多智能体\code\configs\experiments\ablation_top_level_objective.json)
- [configs/experiments/ablation_evolution.json](d:\文件\论文\第二篇论文-etta多智能体\code\configs\experiments\ablation_evolution.json)
- [configs/experiments/ablation_communication.json](d:\文件\论文\第二篇论文-etta多智能体\code\configs\experiments\ablation_communication.json)

Each experiment file contains:

- dataset binding
- metric list
- runtime budget
- paper target values
- optional module switches for ablation
