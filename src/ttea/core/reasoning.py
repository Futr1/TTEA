from __future__ import annotations

from ..config import ModelBackendConfig
from ..models import LangChainPromptBuilder, TorchCommunicationProjector, TransformersTextBackend
from ..types import ReasoningTrace, SystemState, TaskSpec


class ReasoningEngine:
    def __init__(self, config: ModelBackendConfig) -> None:
        self.config = config
        self.prompts = LangChainPromptBuilder(config)
        self.text_backend = TransformersTextBackend(config)
        self.projector = TorchCommunicationProjector(config)

    def prepare(
        self,
        agent_id: str,
        agent_role: str,
        task: TaskSpec,
        state: SystemState,
    ) -> ReasoningTrace:
        prompt = self.prompts.build(agent_id, agent_role, task, state)
        tokens = self.text_backend.tokenize(prompt)
        numeric_features = [
            task.priority,
            task.complexity,
            state.resources.system_load,
            state.resources.resource_pressure,
            state.resources.backlog_depth,
            state.resources.blocked_ratio,
        ]
        projection = self.projector.project(tokens.token_ids, numeric_features)
        keywords = [token for token in tokens.tokens[: min(8, len(tokens.tokens))] if token]
        confidence_bias = min(0.15, max(0.0, len(keywords) / 100.0))
        return ReasoningTrace(
            prompt=prompt,
            token_count=tokens.token_count,
            latent_summary=projection.values[: min(8, len(projection.values))],
            confidence_bias=confidence_bias,
            keywords=keywords,
            metadata={
                "prompt_backend": "langchain" if self.prompts.available else "fallback",
                "tokenizer_backend": tokens.backend,
                "projector_backend": projection.backend,
            },
        )
