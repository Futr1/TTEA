from __future__ import annotations

from dataclasses import asdict, replace

from ..agents import TopologyFactory
from ..config import ExperimentConfig, PlatformConfig
from ..datasets import DatasetLoaderFactory, DatasetRegistry
from ..evaluation import BenchmarkEvaluator
from ..exceptions import TrainingError
from ..execution import TaskExecutionEngine
from ..persistence import ExperimentArtifactStore
from ..runtime import TTEASystem
from ..tasks import KnowledgeTaskAdapter, SoftwareEngineeringTaskAdapter, TranslationTaskAdapter, WebNavigationTaskAdapter
from ..training import HFTrainingService
from ..types import TaskGroup


class BaseExperimentRunner:
    def __init__(self, platform_config: PlatformConfig, experiment_config: ExperimentConfig, registry: DatasetRegistry) -> None:
        self.platform_config = platform_config
        self.experiment_config = experiment_config
        self.registry = registry
        descriptor = registry.get(experiment_config.dataset)
        self.dataset_descriptor = replace(
            descriptor,
            local_path=platform_config.resolve_project_path(experiment_config.dataset_path),
        )
        self.loader = DatasetLoaderFactory.create(self.dataset_descriptor)
        self.adapter = self._select_adapter()
        self.evaluator = BenchmarkEvaluator(experiment_config)
        self.artifact_store = ExperimentArtifactStore(platform_config, experiment_config)

    def _select_adapter(self):
        if self.experiment_config.task_group == TaskGroup.WEB_NAVIGATION:
            return WebNavigationTaskAdapter()
        if self.experiment_config.task_group == TaskGroup.SOFTWARE_ENGINEERING:
            return SoftwareEngineeringTaskAdapter()
        if self.experiment_config.task_group == TaskGroup.TRANSLATION:
            return TranslationTaskAdapter()
        return KnowledgeTaskAdapter()

    def plan(self) -> dict[str, object]:
        topology = TopologyFactory(self.platform_config).build(self.experiment_config.task_group).describe()
        payload = {
            "experiment": self.experiment_config.name,
            "task_group": self.experiment_config.task_group.value,
            "dataset": self.dataset_descriptor.to_dict(),
            "dataset_config_path": str(self.platform_config.resolve_project_path(self.experiment_config.dataset_path)),
            "dataset_available": self.loader.is_available(),
            "metrics": list(self.experiment_config.metrics),
            "runtime": asdict(self.experiment_config.runtime),
            "environment": asdict(self.experiment_config.environment),
            "training": asdict(self.experiment_config.training),
            "persistence": asdict(self.experiment_config.persistence),
            "paper_targets": dict(self.experiment_config.paper_targets),
            "module_switches": {
                "top_level_objective": True,
                "evolution": True,
                "communication": True,
                **self.experiment_config.extras.get("module_switches", {}),
            },
            "model_backend": {
                "provider": self.platform_config.models.provider,
                "model_name_or_path": self.platform_config.models.model_name_or_path,
                "tokenizer_name_or_path": self.platform_config.models.tokenizer_name_or_path,
                "use_torch": self.platform_config.models.use_torch,
                "use_transformers": self.platform_config.models.use_transformers,
                "use_langchain": self.platform_config.models.use_langchain,
            },
            "topology": topology,
        }
        payload.update(self.group_specific_plan())
        return payload

    def preview_tasks(self, limit: int = 2) -> list[dict[str, object]]:
        if self.loader.is_available():
            records = self._load_records(limit=limit, split="dev")
            tasks = [self.adapter.build_task(record, index, self.experiment_config) for index, record in enumerate(records)]
        else:
            tasks = self.adapter.placeholder_tasks(self.experiment_config.dataset, limit, self.experiment_config)
        return [
            {
                "task_id": task.task_id,
                "title": task.title,
                "dataset_name": task.dataset_name,
                "capability_tags": task.capability_tags,
                "priority": task.priority,
                "complexity": task.complexity,
            }
            for task in tasks
        ]

    def run(self, split: str = "test", limit: int | None = None, allow_placeholder: bool = False) -> dict[str, object]:
        used_placeholder = False
        if self.loader.is_available():
            records = self._load_records(limit=limit, split=split)
            tasks = [self.adapter.build_task(record, index, self.experiment_config) for index, record in enumerate(records)]
        elif allow_placeholder:
            used_placeholder = True
            tasks = self.adapter.placeholder_tasks(self.experiment_config.dataset, limit or 4, self.experiment_config)
        else:
            self.loader.ensure_available()
            tasks = []
        system = TTEASystem(self.platform_config, self.experiment_config)
        engine = TaskExecutionEngine(system)
        artifacts = engine.execute_tasks(tasks)
        results = [artifact.result for artifact in artifacts]
        metrics = self.evaluator.evaluate(
            tasks=tasks,
            results=results,
            budget=self.experiment_config.runtime.resource_budget,
            artifacts=[artifact.benchmark for artifact in artifacts],
        )
        payload = {
            "experiment": self.experiment_config.name,
            "task_group": self.experiment_config.task_group.value,
            "task_count": len(tasks),
            "used_placeholder_data": used_placeholder,
            "dataset_available": self.loader.is_available(),
            "split": split,
            "metrics": metrics,
            "paper_targets": self.experiment_config.paper_targets,
            "topology": system.describe_topology(),
            "maintenance_log": system.maintenance_log,
        }
        if self.experiment_config.persistence.enabled:
            run_dir = self.artifact_store.create_run_directory("run")
            prediction_rows = [
                {
                    "task_id": artifact.task.task_id,
                    "dataset": artifact.task.dataset_name,
                    "prediction": artifact.result.response,
                    "success": artifact.result.success,
                    "metrics": artifact.result.metrics,
                    "metadata": artifact.result.metadata,
                    "benchmark": artifact.benchmark,
                }
                for artifact in artifacts
            ]
            trace_rows = [
                {
                    "task_id": artifact.task.task_id,
                    "trace": artifact.trace,
                }
                for artifact in artifacts
            ]
            artifact_paths = self.artifact_store.persist_experiment_run(
                run_dir=run_dir,
                plan=self.plan(),
                payload=payload,
                predictions=prediction_rows,
                traces=trace_rows,
            )
            payload["run_dir"] = str(run_dir)
            payload["artifacts"] = artifact_paths
        return payload

    def train(self, allow_placeholder: bool = False) -> dict[str, object]:
        if not self.experiment_config.training.enabled:
            raise TrainingError(f"Training is disabled for experiment {self.experiment_config.name}.")
        if self.loader.is_available():
            train_records = self._load_records(
                limit=self.experiment_config.training.max_train_samples,
                split=self.experiment_config.training.train_split,
            )
            eval_records = self._load_records(
                limit=self.experiment_config.training.max_eval_samples,
                split=self.experiment_config.training.eval_split,
            )
            train_tasks = [
                self.adapter.build_task(record, index, self.experiment_config)
                for index, record in enumerate(train_records)
            ]
            eval_tasks = [
                self.adapter.build_task(record, index, self.experiment_config)
                for index, record in enumerate(eval_records)
            ]
            used_placeholder = False
        elif allow_placeholder:
            used_placeholder = True
            train_tasks = self.adapter.placeholder_tasks(
                self.experiment_config.dataset,
                self.experiment_config.training.max_train_samples,
                self.experiment_config,
            )
            eval_tasks = self.adapter.placeholder_tasks(
                self.experiment_config.dataset,
                self.experiment_config.training.max_eval_samples,
                self.experiment_config,
            )
        else:
            self.loader.ensure_available()
            train_tasks = []
            eval_tasks = []
            used_placeholder = False
        trainer = HFTrainingService(self.platform_config, self.experiment_config)
        run_dir = self.artifact_store.create_run_directory("train")
        training_output = run_dir / self.experiment_config.training.output_subdir
        artifact = trainer.train(train_tasks, eval_tasks, output_dir=training_output)
        payload = {
            "experiment": self.experiment_config.name,
            "task_group": self.experiment_config.task_group.value,
            "dataset": self.experiment_config.dataset,
            "used_placeholder_data": used_placeholder,
            "train_task_count": len(train_tasks),
            "eval_task_count": len(eval_tasks),
            "training": artifact.summary,
            "checkpoint_index": artifact.checkpoint_index_path,
            "run_dir": str(run_dir),
        }
        if self.experiment_config.persistence.enabled:
            artifact_paths = self.artifact_store.persist_training_run(
                run_dir=run_dir,
                payload=payload,
                history_rows=artifact.history,
            )
            payload["artifacts"] = artifact_paths
        return payload

    def _load_records(self, limit: int | None, split: str) -> list[dict[str, object]]:
        lowered = self.experiment_config.dataset.lower()
        if lowered == "jrc-acquis":
            pair = self.experiment_config.extras.get("language_pairs", ["en-de"])[0]
            source_language, target_language = pair.split("-")
            return self.loader.load_records(
                split=split,
                limit=limit,
                source_language=source_language,
                target_language=target_language,
            )
        if lowered == "squad":
            return self.loader.load_records(split=split, limit=limit)
        return self.loader.load_records(split=split, limit=limit)

    def group_specific_plan(self) -> dict[str, object]:
        return {}


class WebNavigationRunner(BaseExperimentRunner):
    def group_specific_plan(self) -> dict[str, object]:
        return {
            "execution_profile": {
                "environment_type": "browser_or_simulated_web",
                "coordination_focus": ["navigation", "verification"],
                "expected_domains": ["CMS", "Map", "Shop", "red", "Git"],
            }
        }


class TranslationRunner(BaseExperimentRunner):
    def group_specific_plan(self) -> dict[str, object]:
        return {
            "execution_profile": {
                "environment_type": "parallel_text",
                "coordination_focus": ["translation", "quality_assurance"],
                "language_pairs": self.experiment_config.extras.get("language_pairs", []),
            }
        }


class SoftwareEngineeringRunner(BaseExperimentRunner):
    def group_specific_plan(self) -> dict[str, object]:
        return {
            "execution_profile": {
                "environment_type": "repository_issue_records",
                "coordination_focus": ["development", "review", "testing"],
                "artifact_targets": ["patch_plan", "review_notes", "test_feedback"],
            }
        }


class KnowledgeRunner(BaseExperimentRunner):
    def group_specific_plan(self) -> dict[str, object]:
        return {
            "execution_profile": {
                "environment_type": "retrieval_and_reasoning",
                "coordination_focus": ["retrieval", "reasoning"],
                "reasoning_modes": ["evidence_collection", "answer_synthesis", "verification"],
            }
        }


def build_runner(platform_config: PlatformConfig, experiment_config: ExperimentConfig) -> BaseExperimentRunner:
    registry = DatasetRegistry(platform_config.paths.data_root, platform_config.root_dir)
    if experiment_config.task_group == TaskGroup.WEB_NAVIGATION:
        return WebNavigationRunner(platform_config, experiment_config, registry)
    if experiment_config.task_group == TaskGroup.SOFTWARE_ENGINEERING:
        return SoftwareEngineeringRunner(platform_config, experiment_config, registry)
    if experiment_config.task_group == TaskGroup.TRANSLATION:
        return TranslationRunner(platform_config, experiment_config, registry)
    return KnowledgeRunner(platform_config, experiment_config, registry)
