from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, PlatformConfig
from ..exceptions import PersistenceError
from ..utils import ensure_directory, write_json_file, write_jsonl_file


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, Enum):
                normalized[key.value] = _json_safe(item)
            else:
                normalized[str(key)] = _json_safe(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


class ExperimentArtifactStore:
    def __init__(self, platform_config: PlatformConfig, experiment_config: ExperimentConfig) -> None:
        self.platform_config = platform_config
        self.experiment_config = experiment_config
        base_output_root = platform_config.resolve_project_path(platform_config.paths.output_root)
        configured_root = Path(experiment_config.persistence.output_subdir or ".")
        if configured_root.is_absolute():
            self.root = ensure_directory(configured_root)
        else:
            self.root = ensure_directory(base_output_root / configured_root)

    def create_run_directory(self, kind: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_dir = self.root / f"{kind}-{self.experiment_config.name}-{timestamp}"
        return ensure_directory(run_dir)

    def persist_experiment_run(
        self,
        run_dir: Path,
        plan: dict[str, Any],
        payload: dict[str, Any],
        predictions: list[dict[str, Any]],
        traces: list[dict[str, Any]],
    ) -> dict[str, str]:
        try:
            output: dict[str, str] = {}
            output["config"] = str(
                write_json_file(
                    run_dir / "config_snapshot.json",
                    {
                        "platform": _json_safe(self.platform_config),
                        "experiment": _json_safe(self.experiment_config),
                    },
                )
            )
            output["plan"] = str(write_json_file(run_dir / "plan.json", _json_safe(plan)))
            output["summary"] = str(write_json_file(run_dir / "summary.json", _json_safe(payload)))
            if self.experiment_config.persistence.save_metrics:
                output["metrics"] = str(write_json_file(run_dir / "metrics.json", _json_safe(payload.get("metrics", {}))))
            if self.experiment_config.persistence.save_predictions:
                output["predictions"] = str(write_jsonl_file(run_dir / "predictions.jsonl", [_json_safe(item) for item in predictions]))
            if self.experiment_config.persistence.save_task_traces:
                output["traces"] = str(write_jsonl_file(run_dir / "task_traces.jsonl", [_json_safe(item) for item in traces]))
            return output
        except Exception as exc:
            raise PersistenceError(f"Failed to persist experiment artifacts in {run_dir}: {exc}") from exc

    def persist_training_run(
        self,
        run_dir: Path,
        payload: dict[str, Any],
        history_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        try:
            output: dict[str, str] = {}
            output["config"] = str(
                write_json_file(
                    run_dir / "config_snapshot.json",
                    {
                        "platform": _json_safe(self.platform_config),
                        "experiment": _json_safe(self.experiment_config),
                    },
                )
            )
            output["summary"] = str(write_json_file(run_dir / "training_summary.json", _json_safe(payload)))
            if self.experiment_config.persistence.save_training_history and history_rows is not None:
                output["history"] = str(write_jsonl_file(run_dir / "training_history.jsonl", [_json_safe(item) for item in history_rows]))
            return output
        except Exception as exc:
            raise PersistenceError(f"Failed to persist training artifacts in {run_dir}: {exc}") from exc
