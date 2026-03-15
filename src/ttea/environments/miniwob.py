from __future__ import annotations

from typing import Any

from ..config import EnvironmentConfig
from ..exceptions import EnvironmentIntegrationError
from ..integrations import import_gymnasium
from ..types import TaskSpec
from .base import EnvironmentAction, EnvironmentObservation, EnvironmentStep, WebEnvironmentAdapter


class MiniWoBEnvironmentAdapter(WebEnvironmentAdapter):
    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        self._gymnasium = import_gymnasium()
        self._env = None
        self._last_observation = None
        self._last_reward = 0.0

    def _ensure_env(self, task: TaskSpec) -> None:
        if self._env is not None:
            return
        if self._gymnasium is None:
            raise EnvironmentIntegrationError("gymnasium is not installed.")
        env_id = str(task.metadata.get("env_id") or self.config.env_id)
        if not env_id:
            raise EnvironmentIntegrationError("MiniWoB task does not define env_id and no default env_id is configured.")
        try:
            self._env = self._gymnasium.make(env_id)
        except Exception as exc:
            raise EnvironmentIntegrationError(f"Failed to create MiniWoB environment {env_id}: {exc}") from exc

    def reset(self, task: TaskSpec) -> EnvironmentObservation:
        self._ensure_env(task)
        observation, info = self._env.reset()
        self._last_observation = observation
        return self._normalize_observation(observation, info)

    def step(self, action: EnvironmentAction) -> EnvironmentStep:
        if self._env is None:
            raise EnvironmentIntegrationError("MiniWoB environment is not initialized.")
        env_action = self._build_env_action(action)
        observation, reward, terminated, truncated, info = self._env.step(env_action)
        self._last_observation = observation
        self._last_reward = float(reward)
        normalized = self._normalize_observation(observation, info)
        return EnvironmentStep(
            observation=normalized,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    def _build_env_action(self, action: EnvironmentAction) -> Any:
        if hasattr(self._env, "create_action"):
            return self._env.create_action(action.action_type, selector=action.selector, text=action.text, value=action.value)
        if action.action_type == "type":
            return {"type": "text", "text": action.text or action.value}
        if action.action_type == "click":
            return {"type": "click", "selector": action.selector, "value": action.value}
        if action.action_type == "select":
            return {"type": "select", "selector": action.selector, "value": action.value}
        if action.action_type == "wait":
            return {"type": "wait"}
        return {"type": action.action_type, "selector": action.selector, "text": action.text, "value": action.value}

    def _normalize_observation(self, observation: Any, info: dict[str, Any]) -> EnvironmentObservation:
        if isinstance(observation, dict):
            content = observation.get("utterance") or observation.get("dom") or observation.get("text") or str(observation)
            title = observation.get("task_name") or info.get("task_name", "MiniWoB++")
            url = observation.get("url") or info.get("url", "")
        else:
            content = str(observation)
            title = info.get("task_name", "MiniWoB++")
            url = info.get("url", "")
        return EnvironmentObservation(
            url=str(url),
            title=str(title),
            content=str(content)[: self.config.observation_max_chars],
            metadata={"raw_info": info},
        )

    def evaluate(self, task: TaskSpec, trajectory: list[EnvironmentAction]) -> dict[str, Any]:
        success = self._last_reward >= self.config.success_reward
        return {
            "success": success,
            "benchmark_success": success,
            "reward": self._last_reward,
            "trajectory_length": len(trajectory),
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
