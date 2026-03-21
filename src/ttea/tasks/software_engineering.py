from __future__ import annotations

from ..config import ExperimentConfig
from ..types import TaskGroup, TaskSpec
from .base import BaseTaskAdapter


def _normalize_lines(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        return [line.strip() for line in raw.splitlines() if line.strip()]
    return []


class SoftwareEngineeringTaskAdapter(BaseTaskAdapter):
    def build_task(self, record: dict[str, object], index: int, experiment: ExperimentConfig) -> TaskSpec:
        task_id = str(record.get("instance_id") or record.get("id") or f"{experiment.name}-{index}")
        repo = str(record.get("repo") or record.get("repository") or record.get("project") or "unknown-repo")
        title = str(record.get("title") or record.get("issue_title") or f"{repo} issue {index}")
        issue_text = str(record.get("problem_statement") or record.get("issue") or record.get("description") or title)
        hints_text = str(record.get("hints_text") or record.get("hint") or "").strip()
        failing_tests = _normalize_lines(record.get("FAIL_TO_PASS") or record.get("failing_tests") or record.get("tests"))
        regression_tests = _normalize_lines(record.get("PASS_TO_PASS") or record.get("regression_tests"))
        reference_patch = str(record.get("patch") or record.get("gold_patch") or record.get("reference_patch") or "").strip()
        test_command = str(record.get("test_command") or record.get("command") or "pytest -q").strip()
        capability_tags = list(record.get("capability_tags", ["development", "review", "testing"]))

        details = [issue_text]
        if hints_text:
            details.append(f"Hints: {hints_text}")
        if failing_tests:
            details.append(f"Failing tests: {'; '.join(failing_tests[:6])}")
        if regression_tests:
            details.append(f"Regression tests: {'; '.join(regression_tests[:6])}")

        return TaskSpec(
            task_id=task_id,
            title=title[:80],
            description="\n\n".join(details),
            group=TaskGroup.SOFTWARE_ENGINEERING,
            dataset_name=experiment.dataset,
            capability_tags=capability_tags,
            priority=0.78,
            complexity=min(1.0, 0.45 + len(issue_text.split()) / 140.0 + len(failing_tests) * 0.03),
            metadata={
                "repo": repo,
                "base_commit": str(record.get("base_commit") or record.get("commit") or ""),
                "issue_text": issue_text,
                "hints_text": hints_text,
                "failing_tests": failing_tests,
                "regression_tests": regression_tests,
                "test_command": test_command,
                "reference_text": reference_patch or "resolved",
                "reference_answer": reference_patch or "resolved",
                "reference_patch": reference_patch,
            },
        )

    def placeholder_tasks(self, dataset_name: str, limit: int, experiment: ExperimentConfig) -> list[TaskSpec]:
        records = [
            {
                "instance_id": f"{dataset_name.lower().replace(' ', '_')}-{index}",
                "repo": "placeholder/repo",
                "title": f"Placeholder software issue {index}",
                "problem_statement": f"Repair the failing behavior described in placeholder issue {index}.",
                "FAIL_TO_PASS": [f"tests/test_case_{index}.py::test_fix"],
                "PASS_TO_PASS": [f"tests/test_regression_{index}.py::test_existing_behavior"],
                "test_command": "pytest -q",
                "patch": "diff --git a/example.py b/example.py",
            }
            for index in range(limit)
        ]
        return [self.build_task(record, index, experiment) for index, record in enumerate(records)]
