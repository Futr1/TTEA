from __future__ import annotations

import json
import math
from collections import Counter

from ..types import TaskExecutionResult, TaskSpec
from ..utils.text import cosine_from_counters, longest_common_subsequence, ngram_counts, normalize_text, safe_divide, tokenize


def _extract_answer_text(response: str) -> str:
    response = response.strip()
    if not response:
        return ""
    if response.startswith("{"):
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return response
        if isinstance(payload, dict):
            return str(payload.get("answer") or payload.get("status") or payload)
    return response


def success_rate(results: list[TaskExecutionResult]) -> float:
    return safe_divide(sum(1 for result in results if result.success), max(1, len(results))) * 100.0


def accuracy(predictions: list[str], references: list[str]) -> float:
    matches = 0
    for prediction, reference in zip(predictions, references, strict=False):
        if normalize_text(_extract_answer_text(prediction)) == normalize_text(reference):
            matches += 1
    return safe_divide(matches, max(1, len(references))) * 100.0


def string_exact_match(predictions: list[str], references: list[str]) -> float:
    return accuracy(predictions, references)


def bleu(predictions: list[str], references: list[str], max_order: int = 4) -> float:
    precisions: list[float] = []
    prediction_length = 0
    reference_length = 0
    for order in range(1, max_order + 1):
        matches = 0
        candidates = 0
        for prediction, reference in zip(predictions, references, strict=False):
            pred_tokens = tokenize(_extract_answer_text(prediction))
            ref_tokens = tokenize(reference)
            prediction_length += len(pred_tokens) if order == 1 else 0
            reference_length += len(ref_tokens) if order == 1 else 0
            pred_counts = ngram_counts(pred_tokens, order)
            ref_counts = ngram_counts(ref_tokens, order)
            overlap = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
            matches += overlap
            candidates += max(0, len(pred_tokens) - order + 1)
        precisions.append(safe_divide(matches, max(1, candidates)))
    clipped = [max(precision, 1e-9) for precision in precisions]
    brevity_penalty = 1.0
    if prediction_length < reference_length and prediction_length > 0:
        brevity_penalty = math.exp(1.0 - (reference_length / prediction_length))
    score = brevity_penalty * math.exp(sum(math.log(value) for value in clipped) / max_order)
    return score * 100.0


def rouge_l(predictions: list[str], references: list[str]) -> float:
    scores: list[float] = []
    for prediction, reference in zip(predictions, references, strict=False):
        pred_tokens = tokenize(_extract_answer_text(prediction))
        ref_tokens = tokenize(reference)
        lcs = longest_common_subsequence(pred_tokens, ref_tokens)
        precision = safe_divide(lcs, max(1, len(pred_tokens)))
        recall = safe_divide(lcs, max(1, len(ref_tokens)))
        if precision + recall == 0:
            scores.append(0.0)
            continue
        scores.append((2 * precision * recall) / (precision + recall))
    return safe_divide(sum(scores), max(1, len(scores))) * 100.0


def mauve_proxy(predictions: list[str], references: list[str]) -> float:
    prediction_tokens = Counter(token for prediction in predictions for token in tokenize(_extract_answer_text(prediction)))
    reference_tokens = Counter(token for reference in references for token in tokenize(reference))
    similarity = cosine_from_counters(prediction_tokens, reference_tokens)
    return similarity * 100.0


def resource_usage(results: list[TaskExecutionResult], budget: float) -> float:
    spent = sum(result.resource_spent for result in results)
    return min(100.0, safe_divide(spent, max(1e-9, budget * max(1, len(results)))) * 100.0)


def evaluate_metric_set(
    metric_names: list[str],
    tasks: list[TaskSpec],
    results: list[TaskExecutionResult],
    budget: float,
) -> dict[str, float]:
    predictions = [result.response for result in results]
    references = [str(task.metadata.get("reference_text", "")) for task in tasks]
    registry = {
        "success_rate": lambda: success_rate(results),
        "accuracy": lambda: accuracy(predictions, references),
        "string_exact_match": lambda: string_exact_match(predictions, references),
        "bleu": lambda: bleu(predictions, references),
        "rouge_l": lambda: rouge_l(predictions, references),
        "mauve_proxy": lambda: mauve_proxy(predictions, references),
        "resource_usage": lambda: resource_usage(results, budget),
    }
    return {metric: registry[metric]() for metric in metric_names}
