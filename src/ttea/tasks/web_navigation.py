from __future__ import annotations

from ..config import ExperimentConfig
from ..types import TaskGroup, TaskSpec
from .base import BaseTaskAdapter


class WebNavigationTaskAdapter(BaseTaskAdapter):
    def build_task(self, record: dict[str, object], index: int, experiment: ExperimentConfig) -> TaskSpec:
        title = str(record.get("task") or record.get("goal") or f"web-task-{index}")
        description = str(record.get("description") or title)
        domain = str(record.get("domain") or record.get("website") or "generic")
        steps = int(record.get("steps", 8))
        difficulty = float(record.get("difficulty", min(1.0, steps / 16.0)))
        capability_tags = list(record.get("capability_tags", ["navigation", "verification"]))
        evaluation = record.get("evaluation", {})
        if not isinstance(evaluation, dict):
            evaluation = {}
        return TaskSpec(
            task_id=str(record.get("id", f"{experiment.name}-{index}")),
            title=title,
            description=description,
            group=TaskGroup.WEB_NAVIGATION,
            dataset_name=experiment.dataset,
            capability_tags=capability_tags,
            priority=0.65 if domain in {"git", "cms"} else 0.55,
            complexity=difficulty,
            metadata={
                "domain": domain,
                "max_steps": steps,
                "start_url": record.get("start_url") or record.get("url") or experiment.environment.base_url,
                "env_id": record.get("env_id") or experiment.environment.env_id,
                "reference_text": "completed",
                "evaluation": {
                    "expected_url_contains": evaluation.get("expected_url_contains", record.get("expected_url_contains", "")),
                    "required_text": evaluation.get("required_text", record.get("required_text", "")),
                    "success_selectors": evaluation.get("success_selectors", record.get("success_selectors", [])),
                },
                "action_hints": record.get("action_hints", []),
                "instruction": record.get("instruction", description),
            },
        )

    def placeholder_tasks(self, dataset_name: str, limit: int, experiment: ExperimentConfig) -> list[TaskSpec]:
        records = [
            {
                "id": f"{dataset_name.lower()}-{index}",
                "task": f"placeholder web task {index}",
                "domain": "cms",
                "steps": 10,
                "start_url": experiment.environment.base_url,
                "evaluation": {"required_text": "placeholder"},
            }
            for index in range(limit)
        ]
        return [self.build_task(record, index, experiment) for index, record in enumerate(records)]
