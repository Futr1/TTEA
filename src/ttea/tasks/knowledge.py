from __future__ import annotations

from ..config import ExperimentConfig
from ..types import TaskGroup, TaskSpec
from .base import BaseTaskAdapter


def _normalize_choices(raw_choices: object) -> list[dict[str, str]]:
    if isinstance(raw_choices, dict):
        labels = raw_choices.get("label", [])
        texts = raw_choices.get("text", [])
        return [
            {"label": str(label), "text": str(text)}
            for label, text in zip(labels, texts, strict=False)
        ]
    if isinstance(raw_choices, list):
        normalized: list[dict[str, str]] = []
        for index, item in enumerate(raw_choices):
            if isinstance(item, dict):
                normalized.append(
                    {
                        "label": str(item.get("label", chr(ord("A") + index))),
                        "text": str(item.get("text", item.get("value", ""))),
                    }
                )
            else:
                normalized.append({"label": chr(ord("A") + index), "text": str(item)})
        return normalized
    return []


def _reference_answers(record: dict[str, object], answer: str) -> list[str]:
    answers = record.get("answers")
    if isinstance(answers, list):
        return [str(item) for item in answers if str(item)]
    short_answers = record.get("short_answers")
    if isinstance(short_answers, list):
        return [str(item) for item in short_answers if str(item)]
    qa_pairs = record.get("qa_pairs")
    if isinstance(qa_pairs, list):
        extracted: list[str] = []
        for pair in qa_pairs:
            if not isinstance(pair, dict):
                continue
            values = pair.get("short_answers", [])
            if isinstance(values, list):
                extracted.extend(str(item) for item in values if str(item))
        if extracted:
            return extracted
    return [answer] if answer else []


def _resolve_choice_answer(answer: str, choices: list[dict[str, str]]) -> tuple[str, list[str]]:
    normalized_answer = answer.strip().lower()
    for choice in choices:
        label = choice["label"].strip().lower()
        text = choice["text"].strip()
        if normalized_answer and normalized_answer in {label, text.lower()}:
            return text, [choice["label"], text]
    return answer, [answer] if answer else []


class KnowledgeTaskAdapter(BaseTaskAdapter):
    def build_task(self, record: dict[str, object], index: int, experiment: ExperimentConfig) -> TaskSpec:
        question = str(record.get("question") or record.get("claim") or record.get("prompt") or f"knowledge-task-{index}")
        context = str(record.get("context") or record.get("evidence") or "")
        answer = str(record.get("answer") or record.get("label") or record.get("long_answer") or "")
        choices = _normalize_choices(record.get("choices") or record.get("options") or [])
        if choices:
            answer, choice_answers = _resolve_choice_answer(answer, choices)
            reference_answers = choice_answers
        else:
            reference_answers = _reference_answers(record, answer)
        capability_tags = ["retrieval", "reasoning"]
        if experiment.dataset.lower() in {"pubhealth", "arc-challenge"}:
            capability_tags.append("verification")
        return TaskSpec(
            task_id=str(record.get("id", f"{experiment.name}-{index}")),
            title=question[:80],
            description=question,
            group=TaskGroup.KNOWLEDGE_ENHANCEMENT,
            dataset_name=experiment.dataset,
            capability_tags=capability_tags,
            priority=0.70,
            complexity=min(1.0, 0.35 + len(question.split()) / 30.0),
            metadata={
                "context": context,
                "reference_text": answer,
                "reference_answer": answer,
                "reference_answers": reference_answers,
                "choices": choices,
                "short_answers": reference_answers if experiment.dataset.lower() == "asqa" else [],
            },
        )

    def placeholder_tasks(self, dataset_name: str, limit: int, experiment: ExperimentConfig) -> list[TaskSpec]:
        records = [
            {
                "id": f"{dataset_name.lower()}-{index}",
                "question": f"Placeholder question {index}",
                "context": f"Context paragraph {index}",
                "answer": f"Reference answer {index}",
            }
            for index in range(limit)
        ]
        return [self.build_task(record, index, experiment) for index, record in enumerate(records)]
