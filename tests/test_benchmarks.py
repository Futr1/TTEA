from ttea.config import load_experiment_config
from ttea.evaluation import BenchmarkEvaluator, squad_exact_match, squad_f1
from ttea.types import TaskExecutionResult, TaskGroup, TaskSpec


def test_squad_metrics_match_reference_answers() -> None:
    predictions = ["Denver Broncos", "1999"]
    references = [["Denver Broncos", "Broncos"], ["1999"]]
    assert squad_exact_match(predictions, references) == 100.0
    assert squad_f1(predictions, references) == 100.0


def test_benchmark_evaluator_supports_squad_exact_match_and_f1() -> None:
    experiment = load_experiment_config("configs/experiments/squad.json")
    evaluator = BenchmarkEvaluator(experiment)
    tasks = [
        TaskSpec(
            task_id="squad-1",
            title="who won",
            description="who won",
            group=TaskGroup.KNOWLEDGE_ENHANCEMENT,
            dataset_name="SQuAD",
            capability_tags=["retrieval", "reasoning"],
            metadata={
                "reference_text": "Denver Broncos",
                "reference_answers": ["Denver Broncos", "Broncos"],
            },
        )
    ]
    results = [
        TaskExecutionResult(
            success=True,
            response="Denver Broncos",
            used_skills=[],
            reward=0.0,
            resource_spent=0.1,
        )
    ]
    metrics = evaluator.evaluate(tasks, results, budget=10.0)
    assert metrics["exact_match"] == 100.0
    assert metrics["f1"] == 100.0
