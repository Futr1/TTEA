from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ConfigError
from .types import TaskGroup
from .utils import read_json_file, resolve_path


@dataclass(slots=True)
class PathsConfig:
    data_root: str
    config_root: str
    output_root: str
    cache_root: str


@dataclass(slots=True)
class ObjectiveConfig:
    alpha: float
    beta: float
    delta: float
    gamma: float
    stability_floor: float
    resource_budget: float


@dataclass(slots=True)
class AssessmentConfig:
    feature_names: list[str]
    prediction_bias: dict[str, float]
    decision_penalties: dict[str, float]


@dataclass(slots=True)
class EvolutionConfig:
    skill_learning_rate: float
    system_gain: float
    skill_decay: float
    decay_window: int
    elimination_threshold: float
    survival_reward: float
    max_learning_attempts: int


@dataclass(slots=True)
class CommunicationConfig:
    encoder_dim: int
    macro_scale: float
    micro_scale: float
    text_precision: int
    memory_window: int
    feature_grid_size: int
    confidence_threshold: float
    gaussian_smooth: bool
    gaussian_kernel_size: int
    gaussian_sigma: float
    prompt_downsample_ratio: int
    prompt_dropout: float
    prompt_bias: bool
    fusion_mode: str
    fusion_heads: int
    fusion_dropout: float


@dataclass(slots=True)
class DispatchConfig:
    max_decomposition_depth: int
    max_team_size: int
    blocked_retry_limit: int
    rebalance_interval: int


@dataclass(slots=True)
class ModelBackendConfig:
    provider: str
    model_name_or_path: str
    tokenizer_name_or_path: str
    device: str
    dtype: str
    hidden_size: int
    max_prompt_tokens: int
    local_files_only: bool
    trust_remote_code: bool
    use_langchain: bool
    use_transformers: bool
    use_torch: bool
    generation: dict[str, Any]


@dataclass(slots=True)
class AgentConfig:
    agent_id: str
    role: str
    skills: dict[str, float]


@dataclass(slots=True)
class CategoryConfig:
    name: str
    leader_id: str
    agents: list[AgentConfig] = field(default_factory=list)


@dataclass(slots=True)
class TaskGroupConfig:
    categories: list[CategoryConfig] = field(default_factory=list)


@dataclass(slots=True)
class PlatformConfig:
    project_name: str
    paths: PathsConfig
    objective: ObjectiveConfig
    assessment: AssessmentConfig
    evolution: EvolutionConfig
    communication: CommunicationConfig
    dispatch: DispatchConfig
    models: ModelBackendConfig
    task_groups: dict[TaskGroup, TaskGroupConfig]
    root_dir: Path

    def resolve_project_path(self, path: str | Path) -> Path:
        return resolve_path(path, self.root_dir)


@dataclass(slots=True)
class ExperimentRuntimeConfig:
    max_steps: int
    resource_budget: float
    decomposition_depth: int


@dataclass(slots=True)
class EnvironmentConfig:
    enabled: bool
    provider: str
    browser_name: str
    headless: bool
    base_url: str
    env_id: str
    task_timeout_ms: int
    action_delay_ms: int
    observation_max_chars: int
    viewport_width: int
    viewport_height: int
    success_reward: float


@dataclass(slots=True)
class TrainingConfig:
    enabled: bool
    task_type: str
    train_split: str
    eval_split: str
    output_subdir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    logging_steps: int
    save_steps: int
    max_train_samples: int
    max_eval_samples: int
    fp16: bool
    bf16: bool


@dataclass(slots=True)
class PersistenceConfig:
    enabled: bool
    save_predictions: bool
    save_task_traces: bool
    save_metrics: bool
    save_checkpoints: bool
    save_training_history: bool
    output_subdir: str


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    task_group: TaskGroup
    dataset: str
    dataset_path: str
    metrics: list[str]
    runtime: ExperimentRuntimeConfig
    environment: EnvironmentConfig
    training: TrainingConfig
    persistence: PersistenceConfig
    paper_targets: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)


def _require_keys(payload: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing keys in {context}: {joined}")


def _parse_paths(payload: dict[str, Any]) -> PathsConfig:
    _require_keys(payload, ["data_root", "config_root", "output_root", "cache_root"], "paths")
    return PathsConfig(**payload)


def _parse_objective(payload: dict[str, Any]) -> ObjectiveConfig:
    _require_keys(
        payload,
        ["alpha", "beta", "delta", "gamma", "stability_floor", "resource_budget"],
        "objective",
    )
    return ObjectiveConfig(**payload)


def _parse_assessment(payload: dict[str, Any]) -> AssessmentConfig:
    _require_keys(payload, ["feature_names", "prediction_bias", "decision_penalties"], "assessment")
    return AssessmentConfig(**payload)


def _parse_evolution(payload: dict[str, Any]) -> EvolutionConfig:
    _require_keys(
        payload,
        [
            "skill_learning_rate",
            "system_gain",
            "skill_decay",
            "decay_window",
            "elimination_threshold",
            "survival_reward",
            "max_learning_attempts",
        ],
        "evolution",
    )
    return EvolutionConfig(**payload)


def _parse_communication(payload: dict[str, Any]) -> CommunicationConfig:
    defaults = {
        "feature_grid_size": 4,
        "confidence_threshold": 0.35,
        "gaussian_smooth": True,
        "gaussian_kernel_size": 3,
        "gaussian_sigma": 1.0,
        "prompt_downsample_ratio": 4,
        "prompt_dropout": 0.1,
        "prompt_bias": True,
        "fusion_mode": "attention",
        "fusion_heads": 4,
        "fusion_dropout": 0.0,
    }
    merged = {**defaults, **payload}
    _require_keys(merged, ["encoder_dim", "macro_scale", "micro_scale", "text_precision", "memory_window"], "communication")
    return CommunicationConfig(**merged)


def _parse_dispatch(payload: dict[str, Any]) -> DispatchConfig:
    _require_keys(payload, ["max_decomposition_depth", "max_team_size", "blocked_retry_limit", "rebalance_interval"], "dispatch")
    return DispatchConfig(**payload)


def _parse_models(payload: dict[str, Any]) -> ModelBackendConfig:
    _require_keys(
        payload,
        [
            "provider",
            "model_name_or_path",
            "tokenizer_name_or_path",
            "device",
            "dtype",
            "hidden_size",
            "max_prompt_tokens",
            "local_files_only",
            "trust_remote_code",
            "use_langchain",
            "use_transformers",
            "use_torch",
            "generation",
        ],
        "models",
    )
    return ModelBackendConfig(**payload)


def _parse_environment(payload: dict[str, Any] | None) -> EnvironmentConfig:
    data = {
        "enabled": False,
        "provider": "none",
        "browser_name": "chromium",
        "headless": True,
        "base_url": "",
        "env_id": "",
        "task_timeout_ms": 30000,
        "action_delay_ms": 500,
        "observation_max_chars": 4000,
        "viewport_width": 1440,
        "viewport_height": 900,
        "success_reward": 1.0,
    }
    if payload is not None:
        data.update(payload)
    return EnvironmentConfig(**data)


def _parse_training(payload: dict[str, Any] | None) -> TrainingConfig:
    data = {
        "enabled": False,
        "task_type": "text_generation",
        "train_split": "train",
        "eval_split": "dev",
        "output_subdir": "training",
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "num_train_epochs": 1.0,
        "warmup_ratio": 0.05,
        "gradient_accumulation_steps": 1,
        "logging_steps": 10,
        "save_steps": 200,
        "max_train_samples": 200,
        "max_eval_samples": 100,
        "fp16": False,
        "bf16": False,
    }
    if payload is not None:
        data.update(payload)
    return TrainingConfig(**data)


def _parse_persistence(payload: dict[str, Any] | None) -> PersistenceConfig:
    data = {
        "enabled": True,
        "save_predictions": True,
        "save_task_traces": True,
        "save_metrics": True,
        "save_checkpoints": True,
        "save_training_history": True,
        "output_subdir": "experiments",
    }
    if payload is not None:
        data.update(payload)
    return PersistenceConfig(**data)


def _parse_task_group(payload: dict[str, Any]) -> TaskGroupConfig:
    categories: list[CategoryConfig] = []
    for category in payload.get("categories", []):
        categories.append(
            CategoryConfig(
                name=category["name"],
                leader_id=category["leader_id"],
                agents=[AgentConfig(**agent) for agent in category.get("agents", [])],
            )
        )
    return TaskGroupConfig(categories=categories)


def load_platform_config(path: str | Path = "configs/platform.json") -> PlatformConfig:
    resolved_path = resolve_path(path)
    payload = read_json_file(resolved_path)
    _require_keys(
        payload,
        ["project_name", "paths", "objective", "assessment", "evolution", "communication", "dispatch", "models", "task_groups"],
        "platform configuration",
    )
    task_groups = {
        TaskGroup(group_name): _parse_task_group(group_payload)
        for group_name, group_payload in payload["task_groups"].items()
    }
    return PlatformConfig(
        project_name=payload["project_name"],
        paths=_parse_paths(payload["paths"]),
        objective=_parse_objective(payload["objective"]),
        assessment=_parse_assessment(payload["assessment"]),
        evolution=_parse_evolution(payload["evolution"]),
        communication=_parse_communication(payload["communication"]),
        dispatch=_parse_dispatch(payload["dispatch"]),
        models=_parse_models(payload["models"]),
        task_groups=task_groups,
        root_dir=resolved_path.parent.parent.resolve(),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    resolved_path = resolve_path(path)
    payload = read_json_file(resolved_path)
    _require_keys(payload, ["name", "task_group", "dataset", "dataset_path", "metrics", "runtime", "paper_targets"], "experiment configuration")
    runtime = ExperimentRuntimeConfig(**payload["runtime"])
    extras = {
        key: value
        for key, value in payload.items()
        if key not in {"name", "task_group", "dataset", "dataset_path", "metrics", "runtime", "paper_targets", "environment", "training", "persistence"}
    }
    return ExperimentConfig(
        name=payload["name"],
        task_group=TaskGroup(payload["task_group"]),
        dataset=payload["dataset"],
        dataset_path=payload["dataset_path"],
        metrics=list(payload["metrics"]),
        runtime=runtime,
        environment=_parse_environment(payload.get("environment")),
        training=_parse_training(payload.get("training")),
        persistence=_parse_persistence(payload.get("persistence")),
        paper_targets=payload["paper_targets"],
        extras=extras,
    )
