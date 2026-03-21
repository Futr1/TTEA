from ttea.config import load_experiment_config, load_platform_config
from ttea.types import TaskGroup


def test_platform_config_loads() -> None:
    platform = load_platform_config("configs/platform.json")
    assert platform.project_name == "ttea-platform"
    assert TaskGroup.WEB_NAVIGATION in platform.task_groups
    assert TaskGroup.SOFTWARE_ENGINEERING in platform.task_groups
    assert platform.communication.confidence_threshold == 0.35
    assert platform.communication.fusion_mode == "attention"


def test_experiment_config_loads() -> None:
    experiment = load_experiment_config("configs/experiments/webarena.json")
    assert experiment.dataset == "WebArena"
    assert experiment.task_group == TaskGroup.WEB_NAVIGATION


def test_software_engineering_experiment_config_loads() -> None:
    experiment = load_experiment_config("configs/experiments/swebench_lite.json")
    assert experiment.dataset == "SWE-bench Lite"
    assert experiment.task_group == TaskGroup.SOFTWARE_ENGINEERING
