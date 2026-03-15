from __future__ import annotations

from ..config import ExperimentConfig
from ..types import TaskGroup, TaskSpec
from .base import BaseTaskAdapter


class TranslationTaskAdapter(BaseTaskAdapter):
    def build_task(self, record: dict[str, object], index: int, experiment: ExperimentConfig) -> TaskSpec:
        source_language = str(record.get("source_language", "en"))
        target_language = str(record.get("target_language", "de"))
        source_text = str(record.get("source_text", ""))
        target_text = str(record.get("target_text", ""))
        return TaskSpec(
            task_id=str(record.get("id", f"{experiment.name}-{index}")),
            title=f"{source_language}->{target_language}",
            description=source_text,
            group=TaskGroup.TRANSLATION,
            dataset_name=experiment.dataset,
            capability_tags=["translation", "quality_assurance"],
            priority=0.60,
            complexity=min(1.0, max(0.2, len(source_text.split()) / 40.0)),
            metadata={
                "source_language": source_language,
                "target_language": target_language,
                "source_text": source_text,
                "reference_text": target_text,
            },
        )

    def placeholder_tasks(self, dataset_name: str, limit: int, experiment: ExperimentConfig) -> list[TaskSpec]:
        language_pairs = experiment.extras.get("language_pairs", ["en-de"])
        records = []
        for index in range(limit):
            pair = language_pairs[index % len(language_pairs)]
            source_language, target_language = pair.split("-")
            records.append(
                {
                    "id": f"{dataset_name.lower()}-{index}",
                    "source_language": source_language,
                    "target_language": target_language,
                    "source_text": f"Placeholder sentence {index}",
                    "target_text": f"Placeholder translation {index}",
                }
            )
        return [self.build_task(record, index, experiment) for index, record in enumerate(records)]
