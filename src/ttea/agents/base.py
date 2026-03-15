from __future__ import annotations

import json
from dataclasses import dataclass

from ..types import AgentRole, AgentStatus, DecisionType, Observation, ReasoningTrace, SkillSnapshot, SystemState, TaskExecutionResult, TaskGroup, TaskSpec
from ..utils import safe_divide


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True)
class BaseAgent:
    agent_id: str
    category: str
    role: AgentRole
    skills: dict[str, SkillSnapshot]
    state: AgentStatus = AgentStatus.IDLE
    survival_weight: float = 1.0
    long_term_utility: float = 0.0
    learning_attempts: int = 0

    def skill_match(self, required_skills: list[str]) -> float:
        if not required_skills:
            return 0.5
        values = [self.skills.get(skill, SkillSnapshot(name=skill, proficiency=0.0)).proficiency for skill in required_skills]
        return safe_divide(sum(values), len(values))

    def build_observation(self, task: TaskSpec, state: SystemState) -> Observation:
        return Observation(
            summary=f"{self.agent_id} handling {task.title} in {task.dataset_name}",
            numeric_features={
                "priority": task.priority,
                "complexity": task.complexity,
                "system_load": state.resources.system_load,
                "resource_pressure": state.resources.resource_pressure,
            },
            metadata={"agent": self.agent_id, "category": self.category},
        )

    def touch_skills(self, used_skills: list[str], current_step: int) -> None:
        for skill_name in used_skills:
            if skill_name not in self.skills:
                self.skills[skill_name] = SkillSnapshot(name=skill_name, proficiency=0.2, last_used_step=current_step)
            self.skills[skill_name].last_used_step = current_step

    def decay_skills(self, current_step: int, decay_rate: float, decay_window: int) -> None:
        for skill in self.skills.values():
            if current_step - skill.last_used_step >= decay_window:
                skill.proficiency = _clamp(skill.proficiency - decay_rate, 0.05, 1.5)

    def learn_skill(self, skill_name: str, base_proficiency: float) -> None:
        if skill_name in self.skills:
            self.skills[skill_name].proficiency = _clamp(max(self.skills[skill_name].proficiency, base_proficiency), 0.05, 1.5)
        else:
            self.skills[skill_name] = SkillSnapshot(name=skill_name, proficiency=_clamp(base_proficiency, 0.05, 1.5))
        self.learning_attempts += 1


class IndividualAgent(BaseAgent):
    def __init__(self, agent_id: str, category: str, role: AgentRole, skills: dict[str, SkillSnapshot]) -> None:
        super().__init__(agent_id=agent_id, category=category, role=role, skills=skills)

    def execute(
        self,
        task: TaskSpec,
        current_step: int,
        decision: DecisionType,
        support_factor: float = 0.0,
        reasoning_trace: ReasoningTrace | None = None,
    ) -> TaskExecutionResult:
        self.state = AgentStatus.BUSY
        skill_match = self.skill_match(task.capability_tags)
        role_bonus = 0.08 if self.role == AgentRole.GENERALIST else 0.12
        reasoning_boost = 0.0 if reasoning_trace is None else min(0.10, reasoning_trace.confidence_bias + reasoning_trace.token_count / 4000.0)
        decision_bonus = {
            DecisionType.EXECUTE: 0.10,
            DecisionType.ASSIST: 0.06,
            DecisionType.LEARN: 0.02,
            DecisionType.REJECT: -0.30,
        }[decision]
        success_score = (
            skill_match * 0.55
            + support_factor * 0.20
            + role_bonus
            + self.survival_weight * 0.05
            + task.priority * 0.08
            + decision_bonus
            + reasoning_boost
            - task.complexity * 0.35
        )
        success = success_score >= 0.52 and decision != DecisionType.REJECT
        quality = _clamp(success_score, 0.0, 1.0)
        used_skills = task.capability_tags or ["coordination"]
        self.touch_skills(used_skills, current_step)
        resource_spent = 0.20 + task.complexity * 0.45 + len(used_skills) * 0.05 + support_factor * 0.10
        if decision == DecisionType.LEARN:
            resource_spent += 0.15
        reward = quality * task.priority if success else max(-0.10, 0.15 * task.priority - task.complexity * 0.20)
        response = self._render_response(task, success, used_skills)
        evidence = [f"{self.agent_id}:{skill}" for skill in used_skills]
        if reasoning_trace is not None:
            evidence.extend(
                [
                    f"prompt_tokens={reasoning_trace.token_count}",
                    f"keywords={','.join(reasoning_trace.keywords[:4])}",
                ]
            )
        self.state = AgentStatus.IDLE if success else AgentStatus.BLOCKED
        return TaskExecutionResult(
            success=success,
            response=response,
            used_skills=used_skills,
            reward=reward,
            resource_spent=resource_spent,
            evidence=evidence,
            metrics={"quality": quality, "success_score": success_score, "reasoning_boost": reasoning_boost},
            metadata={
                "decision": decision.value,
                "agent_id": self.agent_id,
                "reasoning_prompt_backend": None if reasoning_trace is None else reasoning_trace.metadata.get("prompt_backend"),
                "tokenizer_backend": None if reasoning_trace is None else reasoning_trace.metadata.get("tokenizer_backend"),
            },
        )

    def _render_response(self, task: TaskSpec, success: bool, used_skills: list[str]) -> str:
        if task.group == TaskGroup.TRANSLATION:
            source_text = str(task.metadata.get("source_text", task.description))
            target_language = str(task.metadata.get("target_language", "target"))
            prefix = f"[{target_language}]"
            content = source_text if success else f"incomplete {source_text}"
            return f"{prefix} {content}".strip()
        if task.group == TaskGroup.KNOWLEDGE_ENHANCEMENT:
            payload = {
                "answer": task.metadata.get("reference_answer", task.description) if success else "insufficient evidence",
                "skills": used_skills,
            }
            return json.dumps(payload, ensure_ascii=False)
        if task.group == TaskGroup.WEB_NAVIGATION:
            payload = {
                "goal": task.title,
                "status": "completed" if success else "blocked",
                "actions": used_skills,
            }
            return json.dumps(payload, ensure_ascii=False)
        return task.description
