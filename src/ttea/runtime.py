from __future__ import annotations

from dataclasses import replace

from .agents import AgentTopology, TopologyFactory
from .config import ExperimentConfig, PlatformConfig
from .core import GlobalMemoryPool, GlobalObjective, KnowledgeSynergyEngine, MacroAdapter, MicroAdapter, ObservationEncoder, ReasoningEngine, SystemImpactAssessment, VectorTextBridge
from .dispatch import TaskDispatcher
from .evolution import EvolutionEngine
from .models import TorchImpactNetwork
from .types import AgentStatus, DecisionType, ResourceSnapshot, SystemState, TaskExecutionResult, TaskSpec, UtilityBreakdown


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class TTEASystem:
    def __init__(self, platform_config: PlatformConfig, experiment_config: ExperimentConfig) -> None:
        self.platform_config = platform_config
        self.experiment_config = experiment_config
        self.objective = GlobalObjective(platform_config.objective)
        self.impact_network = TorchImpactNetwork(platform_config.models)
        self.reasoning = ReasoningEngine(platform_config.models)
        self.assessment = SystemImpactAssessment(platform_config.assessment, impact_network=self.impact_network)
        self.encoder = ObservationEncoder(
            platform_config.communication,
            text_backend=self.reasoning.text_backend,
            projector=self.reasoning.projector,
        )
        self.macro_adapter = MacroAdapter(platform_config.communication)
        self.micro_adapter = MicroAdapter(platform_config.communication)
        self.bridge = VectorTextBridge(platform_config.communication)
        self.topology: AgentTopology = TopologyFactory(platform_config).build(experiment_config.task_group)
        self.global_memory = GlobalMemoryPool(window=platform_config.communication.memory_window)
        self.synergy = KnowledgeSynergyEngine(platform_config.communication)
        self.dispatcher = TaskDispatcher(max_team_size=platform_config.dispatch.max_team_size)
        self.evolution = EvolutionEngine(platform_config.evolution)
        self.module_switches = {
            "top_level_objective": True,
            "evolution": True,
            "communication": True,
            **experiment_config.extras.get("module_switches", {}),
        }
        self.state = SystemState(
            step=0,
            utility=UtilityBreakdown(stability=1.0, efficiency=0.60, resource_cost=0.0, task_reward=0.0),
            resources=ResourceSnapshot(system_load=0.10, resource_pressure=0.05, backlog_depth=0.0, blocked_ratio=0.0),
            completed_tasks=0,
            failed_tasks=0,
        )
        self.maintenance_log: list[dict[str, list[str]]] = []

    def describe_topology(self) -> dict[str, object]:
        return self.topology.describe()

    def run_tasks(self, tasks: list[TaskSpec]) -> list[TaskExecutionResult]:
        results: list[TaskExecutionResult] = []
        for task in tasks:
            results.append(self.run_task(task))
        return results

    def run_task(self, task: TaskSpec) -> TaskExecutionResult:
        if self.module_switches["communication"]:
            self.synergy.synchronize(self.topology.global_leader, self.topology.category_leaders, self.global_memory)
        plan = self.dispatcher.plan(self.topology, task, max_depth=self.experiment_config.runtime.decomposition_depth)
        subtask_results: list[TaskExecutionResult] = []

        for assignment in plan.assignments:
            category_name = str(assignment.task.metadata.get("assigned_category", next(iter(self.topology.category_leaders))))
            leader = self.topology.category_leaders[category_name]
            primary = next(member for member in leader.members if member.agent_id == assignment.primary_agent_id)
            reasoning_trace = self.reasoning.prepare(primary.agent_id, primary.role.value, assignment.task, self.state)

            if self.module_switches["communication"]:
                observation = primary.build_observation(assignment.task, self.state)
                encoded = self.encoder.encode(observation)
                macro = self.macro_adapter.apply(encoded)
                micro = self.micro_adapter.apply(macro, primary.agent_id)
                message = self.bridge.encode(micro)
                leader.capability_map.record_observation(f"{primary.agent_id}: {message}")
                self.global_memory.record_message(f"{primary.agent_id}->{leader.agent_id}: {message}")
                self.global_memory.record_message(
                    f"prompt_backend={reasoning_trace.metadata.get('prompt_backend')} tokenizer={reasoning_trace.metadata.get('tokenizer_backend')}"
                )

            skill_match = primary.skill_match(assignment.task.capability_tags)
            collaboration_need = min(
                1.0,
                (assignment.task.complexity + self.state.resources.blocked_ratio + reasoning_trace.confidence_bias) / 2.0,
            )
            if self.module_switches["top_level_objective"]:
                decision = self.assessment.evaluate(self.state, assignment.task, skill_match, collaboration_need)
                selected_decision = decision.best
            else:
                decision = None
                selected_decision = DecisionType.EXECUTE
            if selected_decision == DecisionType.REJECT and assignment.task.priority >= 0.70:
                selected_decision = DecisionType.ASSIST

            if self.module_switches["evolution"] and selected_decision == DecisionType.LEARN:
                self.evolution.learning.apply(primary, assignment.task.capability_tags)

            if selected_decision == DecisionType.REJECT:
                result = TaskExecutionResult(
                    success=False,
                    response="rejected",
                    used_skills=[],
                    reward=-0.10,
                    resource_spent=0.02,
                    evidence=[decision.rationale] if decision is not None else [],
                    metrics={"quality": 0.0},
                    metadata={"decision": selected_decision.value, "agent_id": primary.agent_id},
                )
                primary.state = AgentStatus.BLOCKED
            else:
                support_ids: list[str] = []
                if selected_decision == DecisionType.ASSIST or skill_match < 0.45:
                    support_ids = self.dispatcher.request_assistance(self.topology, assignment)
                    assignment.support_agent_ids = support_ids
                support_factor = self._support_factor(category_name, support_ids, assignment.task)
                result = primary.execute(
                    task=assignment.task,
                    current_step=self.state.step,
                    decision=selected_decision,
                    support_factor=support_factor,
                    reasoning_trace=reasoning_trace,
                )
                if not result.success and not support_ids and selected_decision != DecisionType.REJECT:
                    support_ids = self._cross_category_support(category_name, assignment.task, primary.agent_id)
                    if support_ids:
                        assignment.support_agent_ids = support_ids
                        support_factor = self._support_factor(category_name, support_ids, assignment.task)
                        result = primary.execute(
                            task=assignment.task,
                            current_step=self.state.step,
                            decision=DecisionType.ASSIST,
                            support_factor=support_factor,
                            reasoning_trace=reasoning_trace,
                        )

            before = replace(self.state.utility)
            self._update_state(assignment.task, result)
            after = replace(self.state.utility)
            delta_utility = self.objective.delta(before, after) if self.module_switches["top_level_objective"] else 0.0
            if self.module_switches["evolution"]:
                self.evolution.skill_reinforcement.apply(
                    primary,
                    result.used_skills,
                    local_reward=result.reward,
                    delta_utility=delta_utility,
                    current_step=self.state.step,
                )
            leader.refresh_capability_map()
            subtask_results.append(result)
            self.state.step += 1

        if self.module_switches["evolution"]:
            self.maintenance_log.append(self.evolution.maintain(self.topology.category_leaders, self.state.step))
        return self._aggregate_task(task, subtask_results)

    def _support_factor(self, category_name: str, support_ids: list[str], task: TaskSpec) -> float:
        if not support_ids:
            return 0.0
        values = []
        for support_id in support_ids:
            member = None
            for leader in self.topology.category_leaders.values():
                member = next((item for item in leader.members if item.agent_id == support_id), None)
                if member is not None:
                    break
            if member is None:
                continue
            values.append(member.skill_match(task.capability_tags))
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _cross_category_support(self, current_category: str, task: TaskSpec, requester_id: str) -> list[str]:
        best_support: tuple[float, str] | None = None
        for category_name, leader in self.topology.category_leaders.items():
            if category_name == current_category:
                continue
            for member in leader.members:
                if member.agent_id == requester_id or member.state == AgentStatus.ELIMINATED:
                    continue
                match = member.skill_match(task.capability_tags)
                if best_support is None or match > best_support[0]:
                    best_support = (match, member.agent_id)
        return [] if best_support is None else [best_support[1]]

    def _update_state(self, task: TaskSpec, result: TaskExecutionResult) -> None:
        stability_delta = 0.04 * task.priority if result.success else -0.06 * task.priority
        efficiency_delta = result.metrics.get("quality", 0.0) * 0.05 - task.complexity * 0.015
        resource_delta = result.resource_spent * 0.10
        updated_utility = UtilityBreakdown(
            stability=_clamp(self.state.utility.stability + stability_delta, 0.0, 2.0),
            efficiency=_clamp(self.state.utility.efficiency + efficiency_delta, 0.0, 2.0),
            resource_cost=max(0.0, self.state.utility.resource_cost + resource_delta),
            task_reward=max(0.0, self.state.utility.task_reward + max(result.reward, 0.0)),
        )
        completed = self.state.completed_tasks + int(result.success)
        failed = self.state.failed_tasks + int(not result.success)
        total = max(1, completed + failed)
        updated_resources = ResourceSnapshot(
            system_load=_clamp(self.state.resources.system_load + result.resource_spent * 0.02 - 0.01, 0.0, 1.0),
            resource_pressure=_clamp(self.objective.resource_usage(updated_utility.resource_cost) / 100.0, 0.0, 1.0),
            backlog_depth=max(0.0, self.state.resources.backlog_depth + (0.0 if result.success else 1.0) - 0.25),
            blocked_ratio=failed / total,
        )
        self.state = SystemState(
            step=self.state.step,
            utility=updated_utility,
            resources=updated_resources,
            completed_tasks=completed,
            failed_tasks=failed,
        )

    def _aggregate_task(self, task: TaskSpec, subtask_results: list[TaskExecutionResult]) -> TaskExecutionResult:
        success_count = sum(1 for result in subtask_results if result.success)
        success = success_count == len(subtask_results) if subtask_results else False
        if not success and subtask_results:
            mean_quality = sum(result.metrics.get("quality", 0.0) for result in subtask_results) / len(subtask_results)
            success = mean_quality >= 0.60
        response = "\n".join(result.response for result in subtask_results)
        evidence = [entry for result in subtask_results for entry in result.evidence]
        return TaskExecutionResult(
            success=success,
            response=response,
            used_skills=list({skill for result in subtask_results for skill in result.used_skills}),
            reward=sum(result.reward for result in subtask_results),
            resource_spent=sum(result.resource_spent for result in subtask_results),
            evidence=evidence,
            metrics={
                "quality": sum(result.metrics.get("quality", 0.0) for result in subtask_results) / max(1, len(subtask_results)),
                "subtask_count": float(len(subtask_results)),
            },
            metadata={"task_id": task.task_id, "dataset": task.dataset_name},
        )
