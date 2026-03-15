from ttea.config import load_platform_config
from ttea.core.objective import GlobalObjective
from ttea.types import UtilityBreakdown


def test_global_objective_delta() -> None:
    platform = load_platform_config("configs/platform.json")
    objective = GlobalObjective(platform.objective)
    before = UtilityBreakdown(stability=1.0, efficiency=0.6, resource_cost=0.1, task_reward=0.2)
    after = UtilityBreakdown(stability=1.1, efficiency=0.7, resource_cost=0.1, task_reward=0.3)
    assert objective.delta(before, after) > 0
