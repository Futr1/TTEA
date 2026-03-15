from __future__ import annotations

from dataclasses import dataclass

from ..config import AssessmentConfig
from ..types import DecisionEvaluation, DecisionType, ImpactEstimate, SystemState, TaskSpec


@dataclass(slots=True)
class _CoefficientTable:
    stability: tuple[float, ...]
    efficiency: tuple[float, ...]
    resource: tuple[float, ...]


class SystemImpactAssessment:
    def __init__(self, config: AssessmentConfig, impact_network=None) -> None:
        self.config = config
        self.impact_network = impact_network
        self.coefficients = _CoefficientTable(
            stability=(0.24, -0.18, -0.16, -0.25, 0.31, -0.12, -0.08, -0.27),
            efficiency=(0.16, -0.05, -0.09, -0.13, 0.42, -0.18, 0.24, -0.14),
            resource=(0.05, 0.33, 0.29, 0.38, -0.21, 0.14, 0.22, 0.18),
        )

    def _features(
        self,
        state: SystemState,
        task: TaskSpec,
        skill_match: float,
        collaboration_need: float,
    ) -> list[float]:
        return [
            task.priority,
            task.complexity,
            state.resources.system_load,
            state.resources.resource_pressure,
            skill_match,
            min(1.0, state.resources.backlog_depth / 10.0),
            collaboration_need,
            state.resources.blocked_ratio,
        ]

    def _dot(self, coefficients: tuple[float, ...], features: list[float]) -> float:
        return sum(weight * feature for weight, feature in zip(coefficients, features, strict=True))

    def estimate_impact(
        self,
        decision: DecisionType,
        state: SystemState,
        task: TaskSpec,
        skill_match: float,
        collaboration_need: float,
    ) -> ImpactEstimate:
        features = self._features(state, task, skill_match, collaboration_need)
        if self.impact_network is not None and getattr(self.impact_network, "available", False):
            predicted = self.impact_network.predict(features)
            stability = predicted[0] + self.config.prediction_bias["stability"]
            efficiency = predicted[1] + self.config.prediction_bias["efficiency"]
            resource = predicted[2] + self.config.prediction_bias["resource"]
        else:
            stability = self._dot(self.coefficients.stability, features) + self.config.prediction_bias["stability"]
            efficiency = self._dot(self.coefficients.efficiency, features) + self.config.prediction_bias["efficiency"]
            resource = self._dot(self.coefficients.resource, features) + self.config.prediction_bias["resource"]

        if decision == DecisionType.REJECT:
            stability -= 0.30 * task.priority
            efficiency -= 0.18 * task.priority
            resource -= 0.06
        elif decision == DecisionType.ASSIST:
            stability += 0.12 * collaboration_need
            efficiency += 0.08 * skill_match
            resource += 0.10 + 0.08 * collaboration_need
        elif decision == DecisionType.LEARN:
            stability += 0.05 * task.priority
            efficiency += 0.10 * task.priority
            resource += 0.14 + 0.10 * task.complexity
        else:
            stability += 0.10 * skill_match
            efficiency += 0.12 * skill_match
            resource += 0.08 * task.complexity

        confidence = max(0.2, min(0.95, 0.35 + skill_match * 0.4 - collaboration_need * 0.1))
        return ImpactEstimate(
            stability_delta=stability,
            efficiency_delta=efficiency,
            resource_delta=max(0.0, resource),
            confidence=confidence,
        )

    def evaluate(
        self,
        state: SystemState,
        task: TaskSpec,
        skill_match: float,
        collaboration_need: float,
    ) -> DecisionEvaluation:
        scores: dict[DecisionType, float] = {}
        impacts: dict[DecisionType, ImpactEstimate] = {}
        reward_gains = {
            DecisionType.REJECT: -0.4 * task.priority,
            DecisionType.ASSIST: 0.6 * task.priority,
            DecisionType.LEARN: 0.5 * task.priority,
            DecisionType.EXECUTE: 0.8 * task.priority * max(skill_match, 0.25),
        }
        for decision in DecisionType:
            impact = self.estimate_impact(decision, state, task, skill_match, collaboration_need)
            impacts[decision] = impact
            score = (
                impact.stability_delta * 4.0
                + impact.efficiency_delta * 3.0
                - impact.resource_delta * 2.5
                + reward_gains[decision]
                - self.config.decision_penalties.get(decision.value, 0.0)
            )
            scores[decision] = score

        best = max(scores, key=scores.get)
        rationale = (
            f"reject={scores[DecisionType.REJECT]:.3f}, "
            f"assist={scores[DecisionType.ASSIST]:.3f}, "
            f"learn={scores[DecisionType.LEARN]:.3f}, "
            f"execute={scores[DecisionType.EXECUTE]:.3f}"
        )
        return DecisionEvaluation(
            reject_score=scores[DecisionType.REJECT],
            assist_score=scores[DecisionType.ASSIST],
            learn_score=scores[DecisionType.LEARN],
            execute_score=scores[DecisionType.EXECUTE],
            best=best,
            rationale=rationale,
        )
