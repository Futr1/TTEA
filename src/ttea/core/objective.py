from __future__ import annotations

from dataclasses import replace

from ..config import ObjectiveConfig
from ..types import ImpactEstimate, UtilityBreakdown


class GlobalObjective:
    def __init__(self, config: ObjectiveConfig) -> None:
        self.config = config

    def compute(self, snapshot: UtilityBreakdown) -> float:
        return (
            self.config.alpha * snapshot.stability
            + self.config.beta * snapshot.efficiency
            - self.config.delta * snapshot.resource_cost
            + self.config.gamma * snapshot.task_reward
        )

    def delta(self, before: UtilityBreakdown, after: UtilityBreakdown) -> float:
        return self.compute(after) - self.compute(before)

    def apply_impact(
        self,
        snapshot: UtilityBreakdown,
        impact: ImpactEstimate,
        task_reward_delta: float = 0.0,
    ) -> UtilityBreakdown:
        return replace(
            snapshot,
            stability=snapshot.stability + impact.stability_delta,
            efficiency=snapshot.efficiency + impact.efficiency_delta,
            resource_cost=max(0.0, snapshot.resource_cost + impact.resource_delta),
            task_reward=snapshot.task_reward + task_reward_delta,
        )

    def score_impact(self, impact: ImpactEstimate, task_reward_delta: float = 0.0) -> float:
        return (
            self.config.alpha * impact.stability_delta
            + self.config.beta * impact.efficiency_delta
            - self.config.delta * impact.resource_delta
            + self.config.gamma * task_reward_delta
        )

    def resource_usage(self, resource_spent: float) -> float:
        if self.config.resource_budget <= 0:
            return 0.0
        return min(100.0, (resource_spent / self.config.resource_budget) * 100.0)

    def violates_red_line(self, snapshot: UtilityBreakdown) -> bool:
        return snapshot.stability < self.config.stability_floor or snapshot.resource_cost > self.config.resource_budget
