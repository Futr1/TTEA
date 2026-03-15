from ttea.config import load_platform_config
from ttea.core.reasoning import ReasoningEngine
from ttea.types import ResourceSnapshot, SystemState, TaskGroup, TaskSpec, UtilityBreakdown


def test_reasoning_engine_fallback() -> None:
    platform = load_platform_config("configs/platform.json")
    engine = ReasoningEngine(platform.models)
    task = TaskSpec(
        task_id="demo",
        title="demo task",
        description="Inspect a placeholder task",
        group=TaskGroup.WEB_NAVIGATION,
        dataset_name="WebArena",
        capability_tags=["navigation"],
        priority=0.7,
        complexity=0.5,
    )
    state = SystemState(
        step=0,
        utility=UtilityBreakdown(stability=1.0, efficiency=0.5, resource_cost=0.0, task_reward=0.0),
        resources=ResourceSnapshot(system_load=0.2, resource_pressure=0.1, backlog_depth=0.0, blocked_ratio=0.0),
    )
    trace = engine.prepare("agent_demo", "specialist", task, state)
    assert trace.token_count > 0
    assert len(trace.latent_summary) > 0
