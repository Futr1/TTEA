from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..types import TaskSpec


@dataclass(slots=True)
class EnvironmentObservation:
    url: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class EnvironmentAction:
    action_type: str
    selector: str = ""
    text: str = ""
    value: str = ""
    url: str = ""
    key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "selector": self.selector,
            "text": self.text,
            "value": self.value,
            "url": self.url,
            "key": self.key,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class EnvironmentStep:
    observation: EnvironmentObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": dict(self.info),
        }


class WebEnvironmentAdapter(ABC):
    @abstractmethod
    def reset(self, task: TaskSpec) -> EnvironmentObservation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: EnvironmentAction) -> EnvironmentStep:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, task: TaskSpec, trajectory: list[EnvironmentAction]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
