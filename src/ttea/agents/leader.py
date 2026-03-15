from __future__ import annotations

from dataclasses import dataclass, field

from ..core.memory import CapabilityStateMap
from ..types import AgentRole, AgentStatus, SkillSnapshot, TaskGroup, TaskSpec
from .base import BaseAgent, IndividualAgent


@dataclass(slots=True)
class CategoryLeader(BaseAgent):
    members: list[IndividualAgent] = field(default_factory=list)
    capability_map: CapabilityStateMap = field(default_factory=lambda: CapabilityStateMap("unknown"))

    def __init__(self, agent_id: str, category: str) -> None:
        BaseAgent.__init__(
            self,
            agent_id=agent_id,
            category=category,
            role=AgentRole.CATEGORY_LEADER,
            skills={"coordination": SkillSnapshot(name="coordination", proficiency=0.8)},
            state=AgentStatus.IDLE,
        )
        self.members = []
        self.capability_map = CapabilityStateMap(category_name=category)

    def add_member(self, agent: IndividualAgent) -> None:
        self.members.append(agent)
        self.capability_map.update_agent(agent)

    def refresh_capability_map(self) -> None:
        for member in self.members:
            self.capability_map.update_agent(member)

    def assign_task(self, task: TaskSpec) -> IndividualAgent:
        ranked = self.capability_map.rank_agents(task.capability_tags, limit=max(1, len(self.members)))
        for candidate_id in ranked:
            for member in self.members:
                if member.agent_id == candidate_id and member.state != AgentStatus.ELIMINATED:
                    return member
        return self.members[0]

    def form_temporary_team(self, task: TaskSpec, requester_id: str, max_team_size: int) -> list[IndividualAgent]:
        team: list[IndividualAgent] = []
        ranked = self.capability_map.rank_agents(task.capability_tags, limit=max_team_size + 1)
        for candidate_id in ranked:
            if candidate_id == requester_id:
                continue
            member = next((item for item in self.members if item.agent_id == candidate_id), None)
            if member is None or member.state == AgentStatus.ELIMINATED:
                continue
            team.append(member)
            if len(team) == max_team_size:
                break
        return team

    def summary(self) -> dict[str, object]:
        self.refresh_capability_map()
        return self.capability_map.summary()


@dataclass(slots=True)
class GlobalLeader(BaseAgent):
    cognitive_view: dict[str, object] = field(default_factory=dict)

    def __init__(self, agent_id: str = "global_leader") -> None:
        BaseAgent.__init__(
            self,
            agent_id=agent_id,
            category="system",
            role=AgentRole.GLOBAL_LEADER,
            skills={"orchestration": SkillSnapshot(name="orchestration", proficiency=0.9)},
            state=AgentStatus.IDLE,
        )
        self.cognitive_view = {}

    def decompose_task(
        self,
        task: TaskSpec,
        available_categories: list[str],
        max_depth: int,
    ) -> list[TaskSpec]:
        if max_depth <= 1:
            return [task]
        selected_categories = [category for category in available_categories if category in task.capability_tags]
        if not selected_categories:
            if task.group == TaskGroup.WEB_NAVIGATION:
                selected_categories = [category for category in available_categories if category in {"navigation", "verification"}]
            elif task.group == TaskGroup.TRANSLATION:
                selected_categories = [category for category in available_categories if category in {"translation", "quality_assurance"}]
            else:
                selected_categories = [category for category in available_categories if category in {"retrieval", "reasoning"}]
        if not selected_categories:
            selected_categories = available_categories[:1]

        subtasks: list[TaskSpec] = []
        for index, category in enumerate(selected_categories[:max_depth], start=1):
            subtask_capability_tags = [category]
            subtasks.append(
                TaskSpec(
                    task_id=f"{task.task_id}::subtask::{index}",
                    title=f"{task.title} [{category}]",
                    description=task.description,
                    group=task.group,
                    dataset_name=task.dataset_name,
                    capability_tags=subtask_capability_tags,
                    priority=task.priority,
                    complexity=min(1.0, task.complexity + index * 0.03),
                    metadata={
                        **task.metadata,
                        "assigned_category": category,
                        "parent_task_id": task.task_id,
                        "parent_capability_tags": list(task.capability_tags),
                    },
                )
            )
        return subtasks
