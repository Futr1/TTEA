from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..exceptions import DatasetUnavailableError
from ..types import TaskGroup
from ..utils import read_json_file, resolve_path


@dataclass(slots=True)
class DatasetDescriptor:
    name: str
    task_group: TaskGroup
    local_path: Path
    expected_files: list[str]
    acquisition_url: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "task_group": self.task_group.value,
            "local_path": str(self.local_path),
            "expected_files": list(self.expected_files),
            "acquisition_url": self.acquisition_url,
            "notes": self.notes,
        }


class DatasetRegistry:
    def __init__(self, data_root: str | Path, base_dir: str | Path | None = None) -> None:
        self.data_root = resolve_path(data_root, base_dir)
        if not self.data_root.exists():
            raise DatasetUnavailableError(f"Dataset root does not exist: {self.data_root}")
        self._datasets = self._scan()

    def _scan(self) -> dict[str, DatasetDescriptor]:
        descriptors: dict[str, DatasetDescriptor] = {}
        for manifest_path in sorted(self.data_root.glob("*/manifest.json")):
            payload = read_json_file(manifest_path)
            repository_root = manifest_path.parents[3]
            descriptor = DatasetDescriptor(
                name=payload["name"],
                task_group=TaskGroup(payload["task_group"]),
                local_path=resolve_path(payload["local_path"], repository_root),
                expected_files=list(payload["expected_files"]),
                acquisition_url=payload["acquisition_url"],
                notes=payload["notes"],
            )
            descriptors[descriptor.name.lower()] = descriptor
        return descriptors

    def all(self) -> list[DatasetDescriptor]:
        return sorted(self._datasets.values(), key=lambda item: item.name)

    def get(self, dataset_name: str) -> DatasetDescriptor:
        key = dataset_name.lower()
        if key not in self._datasets:
            available = ", ".join(dataset.name for dataset in self.all())
            raise DatasetUnavailableError(f"Unknown dataset {dataset_name}. Available datasets: {available}")
        return self._datasets[key]

    def by_task_group(self, task_group: TaskGroup) -> list[DatasetDescriptor]:
        return [descriptor for descriptor in self.all() if descriptor.task_group == task_group]
