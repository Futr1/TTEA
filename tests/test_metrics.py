from ttea.evaluation.metrics import accuracy, bleu, rouge_l, success_rate
from ttea.types import TaskExecutionResult


def test_success_rate() -> None:
    results = [
        TaskExecutionResult(success=True, response="a", used_skills=[], reward=0.0, resource_spent=0.1),
        TaskExecutionResult(success=False, response="b", used_skills=[], reward=0.0, resource_spent=0.1),
    ]
    assert success_rate(results) == 50.0


def test_text_metrics() -> None:
    predictions = ["the cat is here"]
    references = ["the cat is here"]
    assert accuracy(predictions, references) == 100.0
    assert bleu(predictions, references) > 99.0
    assert rouge_l(predictions, references) > 99.0
