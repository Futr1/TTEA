from ttea.config import load_experiment_config, load_platform_config
from ttea.experiments.runners import build_runner


def test_runner_can_use_placeholder_tasks() -> None:
    platform = load_platform_config("configs/platform.json")
    experiment = load_experiment_config("configs/experiments/webarena.json")
    experiment.persistence.enabled = False
    runner = build_runner(platform, experiment)
    result = runner.run(limit=2, allow_placeholder=True)
    assert result["task_count"] == 2
    assert result["used_placeholder_data"] is True


def test_software_engineering_runner_can_use_placeholder_tasks() -> None:
    platform = load_platform_config("configs/platform.json")
    experiment = load_experiment_config("configs/experiments/swebench_lite.json")
    experiment.persistence.enabled = False
    runner = build_runner(platform, experiment)
    result = runner.run(limit=2, allow_placeholder=True)
    assert result["task_count"] == 2
    assert result["used_placeholder_data"] is True
