from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from ..types import AgentSnapshot, AgentStatus
from ..utils import safe_divide

if TYPE_CHECKING:
    from ..agents.base import BaseAgent


class CapabilityStateMap:
    def __init__(self, category_name: str, window: int = 12) -> None:
        self.category_name = category_name
        self.window = window
        self.snapshots: dict[str, AgentSnapshot] = {}
        self.observations: deque[str] = deque(maxlen=window)

    def update_agent(self, agent: "BaseAgent") -> None:
        self.snapshots[agent.agent_id] = AgentSnapshot(
            agent_id=agent.agent_id,
            category=agent.category,
            role=agent.role,
            state=agent.state,
            survival_weight=agent.survival_weight,
            long_term_utility=agent.long_term_utility,
            skills={name: skill.proficiency for name, skill in agent.skills.items()},
        )

    def record_observation(self, message: str) -> None:
        self.observations.append(message)

    def rank_agents(self, required_skills: list[str], limit: int = 3) -> list[str]:
        scored: list[tuple[float, str]] = []
        for agent_id, snapshot in self.snapshots.items():
            if snapshot.state == AgentStatus.ELIMINATED:
                continue
            skill_score = safe_divide(
                sum(snapshot.skills.get(skill, 0.0) for skill in required_skills),
                max(1, len(required_skills)),
            )
            availability_bonus = 0.12 if snapshot.state == AgentStatus.IDLE else -0.08
            total = skill_score + snapshot.survival_weight * 0.05 + availability_bonus
            scored.append((total, agent_id))
        scored.sort(reverse=True)
        return [agent_id for _, agent_id in scored[:limit]]

    def summary(self) -> dict[str, Any]:
        active = sum(1 for snapshot in self.snapshots.values() if snapshot.state != AgentStatus.ELIMINATED)
        blocked = sum(1 for snapshot in self.snapshots.values() if snapshot.state == AgentStatus.BLOCKED)
        mean_utility = safe_divide(sum(snapshot.long_term_utility for snapshot in self.snapshots.values()), max(1, len(self.snapshots)))
        return {
            "category": self.category_name,
            "active_agents": active,
            "blocked_agents": blocked,
            "mean_long_term_utility": mean_utility,
            "observations": list(self.observations),
            "snapshots": [asdict(snapshot) for snapshot in self.snapshots.values()],
        }


class GlobalMemoryPool:
    def __init__(self, window: int = 12) -> None:
        self.window = window
        self.category_history: dict[str, deque[dict[str, Any]]] = {}
        self.cross_class_messages: deque[str] = deque(maxlen=window * 2)

    def update_category(self, category_name: str, summary: dict[str, Any]) -> None:
        history = self.category_history.setdefault(category_name, deque(maxlen=self.window))
        history.append(summary)

    def record_message(self, message: str) -> None:
        self.cross_class_messages.append(message)

    def build_global_view(self) -> dict[str, Any]:
        latest_summaries = {
            category: history[-1]
            for category, history in self.category_history.items()
            if history
        }
        return {
            "categories": latest_summaries,
            "messages": list(self.cross_class_messages),
        }
