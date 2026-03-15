from __future__ import annotations

from dataclasses import dataclass

from ..config import PlatformConfig
from ..types import AgentRole, TaskGroup
from .base import IndividualAgent
from .leader import CategoryLeader, GlobalLeader
from ..types import SkillSnapshot


@dataclass(slots=True)
class AgentTopology:
    global_leader: GlobalLeader
    category_leaders: dict[str, CategoryLeader]

    def describe(self) -> dict[str, object]:
        return {
            "global_leader": self.global_leader.agent_id,
            "categories": {
                category: {
                    "leader_id": leader.agent_id,
                    "agents": [member.agent_id for member in leader.members],
                }
                for category, leader in self.category_leaders.items()
            },
        }


class TopologyFactory:
    def __init__(self, config: PlatformConfig) -> None:
        self.config = config

    def build(self, task_group: TaskGroup) -> AgentTopology:
        group_config = self.config.task_groups[task_group]
        global_leader = GlobalLeader()
        category_leaders: dict[str, CategoryLeader] = {}
        for category_config in group_config.categories:
            leader = CategoryLeader(agent_id=category_config.leader_id, category=category_config.name)
            for agent_config in category_config.agents:
                role = AgentRole(agent_config.role)
                skills = {
                    skill_name: SkillSnapshot(name=skill_name, proficiency=value)
                    for skill_name, value in agent_config.skills.items()
                }
                leader.add_member(
                    IndividualAgent(
                        agent_id=agent_config.agent_id,
                        category=category_config.name,
                        role=role,
                        skills=skills,
                    )
                )
            category_leaders[category_config.name] = leader
        return AgentTopology(global_leader=global_leader, category_leaders=category_leaders)
