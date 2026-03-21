from __future__ import annotations

import re
import string
from collections import Counter
from statistics import mean
from typing import Any

from ..config import ExperimentConfig
from ..integrations import import_mauve, import_rouge_score, import_sacrebleu
from ..types import TaskExecutionResult, TaskSpec
from ..utils import longest_common_subsequence, normalize_text, safe_divide, tokenize
from .metrics import bleu as fallback_bleu
from .metrics import mauve_proxy, resource_usage, rouge_l as fallback_rouge_l, success_rate


_ARTICLES_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def _extract_answer_text(response: str) -> str:
    response = response.strip()
    if not response:
        return ""
    if response.startswith("{") and response.endswith("}"):
        try:
            import json

            payload = json.loads(response)
        except Exception:
            return response
        if isinstance(payload, dict):
            return str(payload.get("answer") or payload.get("status") or response)
    return response


def squad_normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    without_articles = _ARTICLES_PATTERN.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def squad_exact_match(predictions: list[str], references: list[list[str]]) -> float:
    scores: list[float] = []
    for prediction, answer_set in zip(predictions, references, strict=False):
        normalized_prediction = squad_normalize_answer(_extract_answer_text(prediction))
        normalized_references = [squad_normalize_answer(answer) for answer in answer_set if answer]
        if not normalized_references:
            scores.append(0.0)
            continue
        scores.append(1.0 if normalized_prediction in normalized_references else 0.0)
    return mean(scores) * 100.0 if scores else 0.0


def squad_f1(predictions: list[str], references: list[list[str]]) -> float:
    scores: list[float] = []
    for prediction, answer_set in zip(predictions, references, strict=False):
        prediction_tokens = squad_normalize_answer(_extract_answer_text(prediction)).split()
        if not answer_set:
            scores.append(0.0)
            continue
        best = 0.0
        for answer in answer_set:
            reference_tokens = squad_normalize_answer(answer).split()
            overlap = Counter(prediction_tokens) & Counter(reference_tokens)
            common = sum(overlap.values())
            if common == 0:
                continue
            precision = safe_divide(common, max(1, len(prediction_tokens)))
            recall = safe_divide(common, max(1, len(reference_tokens)))
            best = max(best, safe_divide(2 * precision * recall, precision + recall))
        scores.append(best)
    return mean(scores) * 100.0 if scores else 0.0


def asqa_string_exact_match(predictions: list[str], short_answers: list[list[str]]) -> float:
    values: list[float] = []
    for prediction, answers in zip(predictions, short_answers, strict=False):
        normalized_prediction = normalize_text(_extract_answer_text(prediction))
        normalized_answers = [normalize_text(answer) for answer in answers if answer]
        if not normalized_answers:
            values.append(0.0)
            continue
        contains_all = all(answer in normalized_prediction for answer in normalized_answers if answer)
        values.append(1.0 if contains_all else 0.0)
    return mean(values) * 100.0 if values else 0.0


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    sacrebleu = import_sacrebleu()
    if sacrebleu is not None:
        return float(sacrebleu.corpus_bleu(predictions, [references]).score)
    return fallback_bleu(predictions, references)


def rouge_l(predictions: list[str], references: list[str]) -> float:
    rouge_score = import_rouge_score()
    if rouge_score is None:
        return fallback_rouge_l(predictions, references)
    scorer = rouge_score.rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(_extract_answer_text(prediction), reference)["rougeL"].fmeasure
        for prediction, reference in zip(predictions, references, strict=False)
    ]
    return mean(scores) * 100.0 if scores else 0.0


def exact_match_any(predictions: list[str], references: list[list[str]]) -> float:
    values: list[float] = []
    for prediction, answer_set in zip(predictions, references, strict=False):
        normalized_prediction = normalize_text(_extract_answer_text(prediction))
        normalized_references = [normalize_text(answer) for answer in answer_set if answer]
        values.append(1.0 if normalized_prediction in normalized_references else 0.0)
    return mean(values) * 100.0 if values else 0.0


def categorical_accuracy(predictions: list[str], references: list[str]) -> float:
    values: list[float] = []
    for prediction, reference in zip(predictions, references, strict=False):
        values.append(1.0 if normalize_text(_extract_answer_text(prediction)) == normalize_text(reference) else 0.0)
    return mean(values) * 100.0 if values else 0.0


def mauve_score(predictions: list[str], references: list[str]) -> float:
    mauve_module = import_mauve()
    if mauve_module is not None:
        try:
            result = mauve_module.compute_mauve(p_text=references, q_text=predictions, verbose=False)
            return float(getattr(result, "mauve", 0.0)) * 100.0
        except Exception:
            pass
    return mauve_proxy(predictions, references)


def token_f1(predictions: list[str], references: list[str]) -> float:
    scores: list[float] = []
    for prediction, reference in zip(predictions, references, strict=False):
        prediction_tokens = tokenize(_extract_answer_text(prediction))
        reference_tokens = tokenize(reference)
        overlap = Counter(prediction_tokens) & Counter(reference_tokens)
        common = sum(overlap.values())
        if common == 0:
            scores.append(0.0)
            continue
        precision = safe_divide(common, max(1, len(prediction_tokens)))
        recall = safe_divide(common, max(1, len(reference_tokens)))
        scores.append(safe_divide(2 * precision * recall, precision + recall))
    return mean(scores) * 100.0 if scores else 0.0


def lcs_recall(predictions: list[str], references: list[str]) -> float:
    values: list[float] = []
    for prediction, reference in zip(predictions, references, strict=False):
        pred_tokens = tokenize(_extract_answer_text(prediction))
        ref_tokens = tokenize(reference)
        lcs = longest_common_subsequence(pred_tokens, ref_tokens)
        values.append(safe_divide(lcs, max(1, len(ref_tokens))))
    return mean(values) * 100.0 if values else 0.0


class BenchmarkEvaluator:
    def __init__(self, experiment_config: ExperimentConfig) -> None:
        self.experiment_config = experiment_config

    def evaluate(
        self,
        tasks: list[TaskSpec],
        results: list[TaskExecutionResult],
        budget: float,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        predictions = [result.response for result in results]
        references = [str(task.metadata.get("reference_text", "")) for task in tasks]
        reference_sets = [
            [str(answer) for answer in task.metadata.get("reference_answers", []) if str(answer)] or [str(task.metadata.get("reference_text", ""))]
            for task in tasks
        ]
        short_answers = [
            [str(answer) for answer in task.metadata.get("short_answers", []) if str(answer)]
            for task in tasks
        ]
        benchmark_rows = artifacts or []
        registry = {
            "success_rate": lambda: success_rate(results),
            "issue_resolution_rate": lambda: self._benchmark_success_rate(results, benchmark_rows),
            "accuracy": lambda: squad_exact_match(predictions, reference_sets)
            if self.experiment_config.dataset.lower() == "squad"
            else categorical_accuracy(predictions, references),
            "bleu": lambda: corpus_bleu([_extract_answer_text(item) for item in predictions], references),
            "rouge_l": lambda: rouge_l(predictions, references),
            "mauve": lambda: mauve_score([_extract_answer_text(item) for item in predictions], references),
            "mauve_proxy": lambda: mauve_score([_extract_answer_text(item) for item in predictions], references),
            "resource_usage": lambda: resource_usage(results, budget),
            "exact_match": lambda: squad_exact_match(predictions, reference_sets),
            "f1": lambda: squad_f1(predictions, reference_sets),
            "string_exact_match": lambda: self._string_exact_match(predictions, reference_sets, short_answers),
            "token_f1": lambda: token_f1(predictions, references),
            "lcs_recall": lambda: lcs_recall(predictions, references),
            "benchmark_success_rate": lambda: self._benchmark_success_rate(results, benchmark_rows),
        }
        return {metric: registry[metric]() for metric in self.experiment_config.metrics}

    def _string_exact_match(
        self,
        predictions: list[str],
        reference_sets: list[list[str]],
        short_answers: list[list[str]],
    ) -> float:
        if self.experiment_config.dataset.lower() == "asqa" and any(short_answers):
            return asqa_string_exact_match(predictions, short_answers)
        return exact_match_any(predictions, reference_sets)

    def _benchmark_success_rate(
        self,
        results: list[TaskExecutionResult],
        artifacts: list[dict[str, Any]],
    ) -> float:
        if not artifacts:
            return success_rate(results)
        values = [
            1.0
            if row.get("benchmark_success") or row.get("success") or row.get("benchmark", {}).get("benchmark_success")
            else 0.0
            for row in artifacts
        ]
        return mean(values) * 100.0 if values else 0.0
