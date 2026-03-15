from __future__ import annotations

from ..config import ModelBackendConfig
from ..integrations import import_langchain_core
from ..types import SystemState, TaskSpec


class LangChainPromptBuilder:
    def __init__(self, config: ModelBackendConfig) -> None:
        self.config = config
        self._langchain = import_langchain_core() if config.use_langchain else None
        self._chat_prompt_template = None
        self._build_template()

    @property
    def available(self) -> bool:
        return self._chat_prompt_template is not None

    def _build_template(self) -> None:
        if self._langchain is None:
            return
        try:
            prompts = __import__("langchain_core.prompts", fromlist=["ChatPromptTemplate"])
            self._chat_prompt_template = prompts.ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a system-level multi-agent worker. Follow the top-level objective and maximize global utility.",
                    ),
                    (
                        "human",
                        "Agent={agent_id}\nRole={agent_role}\nTask={task_title}\nDescription={task_description}\n"
                        "Priority={task_priority}\nComplexity={task_complexity}\nSystemLoad={system_load}\n"
                        "ResourcePressure={resource_pressure}\nBacklogDepth={backlog_depth}\n"
                        "Provide a concise execution plan, risk note, and capability focus.",
                    ),
                ]
            )
        except Exception:
            self._chat_prompt_template = None

    def build(
        self,
        agent_id: str,
        agent_role: str,
        task: TaskSpec,
        state: SystemState,
    ) -> str:
        if self._chat_prompt_template is not None:
            prompt_value = self._chat_prompt_template.invoke(
                {
                    "agent_id": agent_id,
                    "agent_role": agent_role,
                    "task_title": task.title,
                    "task_description": task.description,
                    "task_priority": f"{task.priority:.3f}",
                    "task_complexity": f"{task.complexity:.3f}",
                    "system_load": f"{state.resources.system_load:.3f}",
                    "resource_pressure": f"{state.resources.resource_pressure:.3f}",
                    "backlog_depth": f"{state.resources.backlog_depth:.3f}",
                }
            )
            return prompt_value.to_string()
        return (
            f"[SYSTEM] top_level_utility_first\n"
            f"[AGENT] id={agent_id} role={agent_role}\n"
            f"[TASK] {task.title}\n"
            f"[DESC] {task.description}\n"
            f"[STATE] priority={task.priority:.3f} complexity={task.complexity:.3f} "
            f"system_load={state.resources.system_load:.3f} resource_pressure={state.resources.resource_pressure:.3f}"
        )
