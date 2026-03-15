from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskGroup(str, Enum):
    WEB_NAVIGATION = "web_navigation"
    TRANSLATION = "translation"
    KNOWLEDGE_ENHANCEMENT = "knowledge_enhancement"


class AgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    BLOCKED = "blocked"
    ELIMINATED = "eliminated"


class DecisionType(str, Enum):
    REJECT = "reject"
    ASSIST = "assist"
    LEARN = "learn"
    EXECUTE = "execute"


class AgentRole(str, Enum):
    GLOBAL_LEADER = "global_leader"
    CATEGORY_LEADER = "category_leader"
    SPECIALIST = "specialist"
    GENERALIST = "generalist"


@dataclass(slots=True)
class SkillSnapshot:
    name: str
    proficiency: float
    last_used_step: int = 0
    maintenance_cost: float = 0.05


@dataclass(slots=True)
class Observation:
    summary: str
    numeric_features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    title: str
    description: str
    group: TaskGroup
    dataset_name: str
    capability_tags: list[str]
    priority: float = 0.5
    complexity: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionProposal:
    name: str
    description: str
    required_skills: list[str]
    estimated_cost: float
    predicted_benefit: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UtilityBreakdown:
    stability: float
    efficiency: float
    resource_cost: float
    task_reward: float


@dataclass(slots=True)
class ImpactEstimate:
    stability_delta: float
    efficiency_delta: float
    resource_delta: float
    confidence: float = 0.5


@dataclass(slots=True)
class DecisionEvaluation:
    reject_score: float
    assist_score: float
    learn_score: float
    execute_score: float
    best: DecisionType
    rationale: str


@dataclass(slots=True)
class TaskExecutionResult:
    success: bool
    response: str
    used_skills: list[str]
    reward: float
    resource_spent: float
    evidence: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentSnapshot:
    agent_id: str
    category: str
    role: AgentRole
    state: AgentStatus
    survival_weight: float
    long_term_utility: float
    skills: dict[str, float]


@dataclass(slots=True)
class ResourceSnapshot:
    system_load: float
    resource_pressure: float
    backlog_depth: float
    blocked_ratio: float


@dataclass(slots=True)
class ReasoningTrace:
    prompt: str
    token_count: int
    latent_summary: list[float]
    confidence_bias: float
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SystemState:
    step: int
    utility: UtilityBreakdown
    resources: ResourceSnapshot
    completed_tasks: int = 0
    failed_tasks: int = 0


@dataclass(slots=True)
class TeamAssignment:
    leader_id: str
    primary_agent_id: str
    support_agent_ids: list[str]
    task: TaskSpec
