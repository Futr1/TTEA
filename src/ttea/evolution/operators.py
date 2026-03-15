from __future__ import annotations

from ..config import EvolutionConfig
from ..types import AgentStatus


class SkillReinforcementOperator:
    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config

    def apply(self, agent, used_skills: list[str], local_reward: float, delta_utility: float, current_step: int) -> None:
        bonus = self.config.skill_learning_rate * (local_reward + self.config.system_gain * max(0.0, delta_utility))
        for skill_name in used_skills:
            agent.learn_skill(skill_name, base_proficiency=0.2)
            agent.skills[skill_name].proficiency = max(0.05, min(1.5, agent.skills[skill_name].proficiency + bonus))
            agent.skills[skill_name].last_used_step = current_step
        agent.long_term_utility += delta_utility


class LearningOperator:
    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config

    def apply(self, agent, required_skills: list[str]) -> None:
        if agent.learning_attempts >= self.config.max_learning_attempts:
            return
        missing = [skill for skill in required_skills if skill not in agent.skills]
        for skill_name in missing:
            agent.learn_skill(skill_name, base_proficiency=0.35)
        agent.survival_weight += self.config.survival_reward


class EliminationOperator:
    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config

    def apply(self, category_leader, current_step: int) -> list[str]:
        removed: list[str] = []
        for member in category_leader.members:
            member.decay_skills(current_step, self.config.skill_decay, self.config.decay_window)
            mean_skill = sum(skill.proficiency for skill in member.skills.values()) / max(1, len(member.skills))
            if member.long_term_utility < self.config.elimination_threshold and mean_skill < 0.25:
                member.state = AgentStatus.ELIMINATED
                removed.append(member.agent_id)
        return removed


class EvolutionEngine:
    def __init__(self, config: EvolutionConfig) -> None:
        self.skill_reinforcement = SkillReinforcementOperator(config)
        self.learning = LearningOperator(config)
        self.elimination = EliminationOperator(config)

    def maintain(self, category_leaders: dict[str, object], current_step: int) -> dict[str, list[str]]:
        removed: dict[str, list[str]] = {}
        for category_name, leader in category_leaders.items():
            removed[category_name] = self.elimination.apply(leader, current_step)
        return removed
