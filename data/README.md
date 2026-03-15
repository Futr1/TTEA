# Dataset layout

All dataset folders live under [data/datasets](d:\文件\论文\第二篇论文-etta多智能体\code\data\datasets). Each dataset directory already contains a `manifest.json` file with:

- dataset name
- task group
- expected local files
- acquisition URL
- short acquisition notes

## Local path policy

If a dataset is not available on the machine yet, keep the directory in place and download the files into the path recorded in its manifest.

Expected directories:

- [data/datasets/webarena](d:\文件\论文\第二篇论文-etta多智能体\code\data\datasets\webarena)
- [data/datasets/miniwobpp](d:\文件\论文\第二篇论文-etta多智能体\code\data\datasets\miniwobpp)
- [data/datasets/jrc_acquis](d:\文件\论文\第二篇论文-etta多智能体\code\data\datasets\jrc_acquis)
- [data/datasets/pubhealth](d:\文件\论文\第二篇论文-etta多智能体\code\data\datasets\pubhealth)
- [data/datasets/arc_challenge](d:\文件\论文\第二篇论文-etta多智能体\code\data\datasets\arc_challenge)
- [data/datasets/squad](d:\文件\论文\第二篇论文-etta多智能体\code\data\datasets\squad)
- [data/datasets/asqa](d:\文件\论文\第二篇论文-etta多智能体\code\data\datasets\asqa)

## Suggested acquisition sources

- `WebArena`: official benchmark repository and released task files
- `MiniWoB++`: official benchmark repository or mirrored task suites
- `JRC-Acquis`: official multilingual aligned corpus distribution
- `PubHealth`: public benchmark release
- `ARC-Challenge`: AI2 dataset release
- `SQuAD`: official SQuAD release
- `ASQA`: official project release

The exact URLs are embedded in each manifest so the CLI can surface them later.
