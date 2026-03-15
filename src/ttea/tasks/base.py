from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import ExperimentConfig
from ..types import TaskSpec


class BaseTaskAdapter(ABC):
    @abstractmethod
    def build_task(self, record: dict[str, object], index: int, experiment: ExperimentConfig) -> TaskSpec:
        raise NotImplementedError

    @abstractmethod
    def placeholder_tasks(self, dataset_name: str, limit: int, experiment: ExperimentConfig) -> list[TaskSpec]:
        raise NotImplementedError
