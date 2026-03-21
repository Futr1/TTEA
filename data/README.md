# Dataset layout

All dataset folders live under `data/datasets`. Each dataset directory already contains a `manifest.json` file with:

- dataset name
- task group
- expected local files
- acquisition URL
- short acquisition notes

## Local path policy

If a dataset is not available on the machine yet, keep the directory in place and download the files into the path recorded in its manifest.

Expected directories:

- `data/datasets/webarena`
- `data/datasets/miniwobpp`
- `data/datasets/swebench_lite`
- `data/datasets/jrc_acquis`
- `data/datasets/pubhealth`
- `data/datasets/arc_challenge`
- `data/datasets/squad`
- `data/datasets/asqa`

## Suggested acquisition sources

- `WebArena`: <https://github.com/web-arena-x/webarena>
- `MiniWoB++`: <https://miniwob.farama.org/>
- `SWE-bench Lite`: <https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite>
- `JRC-Acquis`: <https://opus.nlpl.eu/JRC-Acquis.php>
- `PubHealth`: <https://huggingface.co/datasets/health_fact>
- `ARC-Challenge`: <https://allenai.org/data/arc>
- `SQuAD`: <https://rajpurkar.github.io/SQuAD-explorer/>
- `ASQA`: <https://github.com/google-research/language/tree/master/language/asqa>

The exact URLs are embedded in each manifest so the CLI can surface them later.
