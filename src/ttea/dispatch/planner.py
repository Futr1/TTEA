from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import ExecutionBlockedError
from ..types import TeamAssignment, TaskSpec


@dataclass(slots=True)
class DispatchPlan:
    assignments: list[TeamAssignment]

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "leader_id": assignment.leader_id,
                "primary_agent_id": assignment.primary_agent_id,
                "support_agent_ids": assignment.support_agent_ids,
                "task_id": assignment.task.task_id,
                "category": assignment.task.metadata.get("assigned_category"),
            }
            for assignment in self.assignments
        ]


class TaskDispatcher:
    def __init__(self, max_team_size: int) -> None:
        self.max_team_size = max_team_size

    def plan(self, topology, task: TaskSpec, max_depth: int) -> DispatchPlan:
        category_names = list(topology.category_leaders.keys())
        subtasks = topology.global_leader.decompose_task(task, category_names, max_depth=max_depth)
        assignments: list[TeamAssignment] = []
        for subtask in subtasks:
            category_name = str(subtask.metadata.get("assigned_category", category_names[0]))
            leader = topology.category_leaders.get(category_name)
            if leader is None:
                raise ExecutionBlockedError(f"No category leader available for {category_name}")
            primary = leader.assign_task(subtask)
            assignments.append(
                TeamAssignment(
                    leader_id=leader.agent_id,
                    primary_agent_id=primary.agent_id,
                    support_agent_ids=[],
                    task=subtask,
                )
            )
        return DispatchPlan(assignments=assignments)

    def request_assistance(self, topology, assignment: TeamAssignment) -> list[str]:
        category_name = str(assignment.task.metadata.get("assigned_category"))
        leader = topology.category_leaders[category_name]
        team = leader.form_temporary_team(
            task=assignment.task,
            requester_id=assignment.primary_agent_id,
            max_team_size=self.max_team_size,
        )
        return [member.agent_id for member in team]
